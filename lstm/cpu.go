package lstm

import (
	"fmt"
	"math"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/dense"
)

// ForwardCPUTiled — LSTM via Dense gate MatVec.
func ForwardCPUTiled[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	return forwardViaDense(l, input)
}

// BackwardCPUTiled — BPTT; gradW = loom [i|f|g|o] packs.
func BackwardCPUTiled[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	return backwardViaDense(l, gradOut, input, pre)
}

func hostDenseBackends(l *Layer) (restore func()) {
	type snap struct {
		g    *Gate
		ihBe core.Backend
		hhBe core.Backend
	}
	var snaps []snap
	for _, g := range l.gates() {
		snaps = append(snaps, snap{g, g.IH.Exec.Backend, g.HH.Exec.Backend})
		if l.Exec.Backend == core.BackendWebGPU {
			g.IH.Exec.Backend = core.BackendCPUTiled
			g.HH.Exec.Backend = core.BackendCPUTiled
		}
	}
	return func() {
		for _, s := range snaps {
			s.g.IH.Exec.Backend = s.ihBe
			s.g.HH.Exec.Backend = s.hhBe
		}
	}
}

func gateLinear[T core.Numeric](g *Gate, xt, ht *core.Tensor[T]) ([]float64, error) {
	_, ihPost, err := dense.Forward(g.IH, xt)
	if err != nil {
		return nil, err
	}
	_, hhPost, err := dense.Forward(g.HH, ht)
	if err != nil {
		return nil, err
	}
	out := make([]float64, len(ihPost.Data))
	for i := range out {
		out[i] = core.AsFloat64(ihPost.Data[i]) + core.AsFloat64(hhPost.Data[i])
	}
	return out, nil
}

func forwardViaDense[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	lay, err := parseLayout(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	restore := hostDenseBackends(l)
	defer restore()

	pre = core.NewTensor[T](lay.batch, lay.seq, 5*lay.hid)
	post = core.NewTensor[T](lay.batch, lay.seq, lay.hid)
	hPrev := make([]T, lay.batch*lay.hid)
	cPrev := make([]float64, lay.batch*lay.hid)

	for t := 0; t < lay.seq; t++ {
		xt := xAt(input, lay, t)
		ht := hTensor(hPrev, lay.batch, lay.hid)
		iSum, err := gateLinear(l.I, xt, ht)
		if err != nil {
			return nil, nil, fmt.Errorf("lstm i t=%d: %w", t, err)
		}
		fSum, err := gateLinear(l.F, xt, ht)
		if err != nil {
			return nil, nil, fmt.Errorf("lstm f t=%d: %w", t, err)
		}
		gSum, err := gateLinear(l.G, xt, ht)
		if err != nil {
			return nil, nil, fmt.Errorf("lstm g t=%d: %w", t, err)
		}
		oSum, err := gateLinear(l.O, xt, ht)
		if err != nil {
			return nil, nil, fmt.Errorf("lstm o t=%d: %w", t, err)
		}
		for b := 0; b < lay.batch; b++ {
			pBase := b*lay.seq*5*lay.hid + t*5*lay.hid
			for h := 0; h < lay.hid; h++ {
				i := b*lay.hid + h
				iS, fS, gS, oS := iSum[i], fSum[i], gSum[i], oSum[i]
				pre.Data[pBase+h] = core.FromFloat64[T](iS)
				pre.Data[pBase+lay.hid+h] = core.FromFloat64[T](fS)
				pre.Data[pBase+2*lay.hid+h] = core.FromFloat64[T](gS)
				pre.Data[pBase+3*lay.hid+h] = core.FromFloat64[T](oS)

				iG := 1.0 / (1.0 + math.Exp(-iS))
				fG := 1.0 / (1.0 + math.Exp(-fS))
				gG := math.Tanh(gS)
				oG := 1.0 / (1.0 + math.Exp(-oS))
				cC := fG*cPrev[i] + iG*gG
				hC := oG * math.Tanh(cC)

				pre.Data[pBase+4*lay.hid+h] = core.FromFloat64[T](cC)
				post.Data[b*lay.seq*lay.hid+t*lay.hid+h] = core.FromFloat64[T](hC)
				hPrev[i] = core.FromFloat64[T](hC)
				cPrev[i] = cC
			}
		}
	}
	return pre, post, nil
}

func accumulateGate[T core.Numeric](g *Gate, delta, xt, hPrev *core.Tensor[T], dIH, dHH, dB []float64) (gx, gh []float64, err error) {
	ihPre, _, err := dense.Forward(g.IH, xt)
	if err != nil {
		return nil, nil, err
	}
	hhPre, _, err := dense.Forward(g.HH, hPrev)
	if err != nil {
		return nil, nil, err
	}
	gxT, dWIH, err := dense.Backward(g.IH, delta, xt, ihPre)
	if err != nil {
		return nil, nil, err
	}
	ghT, dWHH, err := dense.Backward(g.HH, delta, hPrev, hhPre)
	if err != nil {
		return nil, nil, err
	}
	for i := range dIH {
		dIH[i] += core.AsFloat64(dWIH.Data[i])
	}
	for i := range dHH {
		dHH[i] += core.AsFloat64(dWHH.Data[i])
	}
	hid := delta.Shape[1]
	batch := delta.Shape[0]
	for b := 0; b < batch; b++ {
		for h := 0; h < hid; h++ {
			dB[h] += core.AsFloat64(delta.Data[b*hid+h])
		}
	}
	gx = make([]float64, len(gxT.Data))
	gh = make([]float64, len(ghT.Data))
	for i := range gx {
		gx[i] = core.AsFloat64(gxT.Data[i])
	}
	for i := range gh {
		gh[i] = core.AsFloat64(ghT.Data[i])
	}
	return gx, gh, nil
}

func backwardViaDense[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	lay, err := parseLayout(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	if gradOut == nil || pre == nil {
		return nil, nil, fmt.Errorf("lstm: nil gradOut/pre")
	}
	wantPre := lay.batch * lay.seq * 5 * lay.hid
	wantOut := lay.batch * lay.seq * lay.hid
	if pre.Len() < wantPre || gradOut.Len() < wantOut {
		return nil, nil, fmt.Errorf("lstm: grad/pre short")
	}
	restore := hostDenseBackends(l)
	defer restore()

	ihN := lay.hid * lay.in
	hhN := lay.hid * lay.hid
	gateN := ihN + hhN + lay.hid

	dPack := make([]float64, 4*gateN)
	dIH := [4][]float64{}
	dHH := [4][]float64{}
	dB := [4][]float64{}
	for g := 0; g < 4; g++ {
		base := g * gateN
		dIH[g] = dPack[base : base+ihN]
		dHH[g] = dPack[base+ihN : base+ihN+hhN]
		dB[g] = dPack[base+ihN+hhN : base+gateN]
	}
	gates := []*Gate{l.I, l.F, l.G, l.O}

	gi := make([]float64, lay.batch*lay.seq*lay.in)
	gradH := make([]float64, lay.batch*lay.hid)
	gradC := make([]float64, lay.batch*lay.hid)

	for t := lay.seq - 1; t >= 0; t-- {
		nextGH := make([]float64, lay.batch*lay.hid)
		nextGC := make([]float64, lay.batch*lay.hid)
		deltaT := [4]*core.Tensor[T]{}
		for g := 0; g < 4; g++ {
			deltaT[g] = core.NewTensor[T](lay.batch, lay.hid)
		}

		for b := 0; b < lay.batch; b++ {
			pBase := b*lay.seq*5*lay.hid + t*5*lay.hid
			for h := 0; h < lay.hid; h++ {
				i := b*lay.hid + h
				dh := gradH[i] + core.AsFloat64(gradOut.Data[b*lay.seq*lay.hid+t*lay.hid+h])
				iS := core.AsFloat64(pre.Data[pBase+h])
				fS := core.AsFloat64(pre.Data[pBase+lay.hid+h])
				gS := core.AsFloat64(pre.Data[pBase+2*lay.hid+h])
				oS := core.AsFloat64(pre.Data[pBase+3*lay.hid+h])
				cC := core.AsFloat64(pre.Data[pBase+4*lay.hid+h])
				iG := 1.0 / (1.0 + math.Exp(-iS))
				fG := 1.0 / (1.0 + math.Exp(-fS))
				oG := 1.0 / (1.0 + math.Exp(-oS))
				gG := math.Tanh(gS)
				cT := math.Tanh(cC)
				cP := 0.0
				if t > 0 {
					cP = core.AsFloat64(pre.Data[pBase-5*lay.hid+4*lay.hid+h])
				}
				dc := gradC[i] + dh*oG*(1.0-cT*cT)
				di := dc * gG * iG * (1.0 - iG)
				df := dc * cP * fG * (1.0 - fG)
				dg := dc * iG * (1.0 - gG*gG)
				do := dh * cT * oG * (1.0 - oG)
				deltaT[0].Data[i] = core.FromFloat64[T](di)
				deltaT[1].Data[i] = core.FromFloat64[T](df)
				deltaT[2].Data[i] = core.FromFloat64[T](dg)
				deltaT[3].Data[i] = core.FromFloat64[T](do)
				nextGC[i] = dc * fG
			}
		}

		xt := xAt(input, lay, t)
		var hPrev *core.Tensor[T]
		if t == 0 {
			hPrev = core.NewTensor[T](lay.batch, lay.hid)
		} else {
			hPrev = core.NewTensor[T](lay.batch, lay.hid)
			for b := 0; b < lay.batch; b++ {
				pP := b*lay.seq*5*lay.hid + (t-1)*5*lay.hid
				for h := 0; h < lay.hid; h++ {
					oS := core.AsFloat64(pre.Data[pP+3*lay.hid+h])
					cC := core.AsFloat64(pre.Data[pP+4*lay.hid+h])
					oG := 1.0 / (1.0 + math.Exp(-oS))
					hPrev.Data[b*lay.hid+h] = core.FromFloat64[T](oG * math.Tanh(cC))
				}
			}
		}

		for g := 0; g < 4; g++ {
			gx, gh, err := accumulateGate(gates[g], deltaT[g], xt, hPrev, dIH[g], dHH[g], dB[g])
			if err != nil {
				return nil, nil, fmt.Errorf("lstm gate %d bwd t=%d: %w", g, t, err)
			}
			for b := 0; b < lay.batch; b++ {
				for i := 0; i < lay.in; i++ {
					gi[b*lay.seq*lay.in+t*lay.in+i] += gx[b*lay.in+i]
				}
				for h := 0; h < lay.hid; h++ {
					nextGH[b*lay.hid+h] += gh[b*lay.hid+h]
				}
			}
		}
		gradH, gradC = nextGH, nextGC
	}

	gradIn = core.NewTensor[T](lay.batch, lay.seq, lay.in)
	for i := range gradIn.Data {
		gradIn.Data[i] = core.FromFloat64[T](gi[i])
	}
	gradW = core.NewTensor[T](1, l.GradWSize())
	for i := range dPack {
		gradW.Data[i] = core.FromFloat64[T](dPack[i])
	}
	return gradIn, gradW, nil
}
