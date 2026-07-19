package rnn

import (
	"fmt"
	"math"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
)

// ForwardCPUTiled — sequential RNN via Dense IH/HH MatVec.
func ForwardCPUTiled[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	return forwardViaDense(l, input)
}

// BackwardCPUTiled — BPTT; gradW = [dIH | dHH | dBias].
func BackwardCPUTiled[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	return backwardViaDense(l, gradOut, input, pre)
}

func forwardViaDense[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	lay, err := parseLayout(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	// WebGPU: device required (checked by ForwardWebGPU); recurrence ALU on host today
	// (same honesty bar as RMSNorm/LayerNorm — no fused on-device RNN shader yet).
	ihBe, hhBe := l.IH.Exec.Backend, l.HH.Exec.Backend
	if l.Exec.Backend == core.BackendWebGPU {
		l.IH.Exec.Backend = core.BackendCPUTiled
		l.HH.Exec.Backend = core.BackendCPUTiled
	}
	defer func() {
		l.IH.Exec.Backend = ihBe
		l.HH.Exec.Backend = hhBe
	}()

	pre = core.NewTensor[T](lay.batch, lay.seq, lay.hid)
	post = core.NewTensor[T](lay.batch, lay.seq, lay.hid)
	hPrev := make([]T, lay.batch*lay.hid) // zeros

	for t := 0; t < lay.seq; t++ {
		xt := xAt(input, lay, t)
		ht := hTensor(hPrev, lay.batch, lay.hid)
		_, ihPost, err := dense.Forward(l.IH, xt)
		if err != nil {
			return nil, nil, fmt.Errorf("rnn IH t=%d: %w", t, err)
		}
		_, hhPost, err := dense.Forward(l.HH, ht)
		if err != nil {
			return nil, nil, fmt.Errorf("rnn HH t=%d: %w", t, err)
		}
		for b := 0; b < lay.batch; b++ {
			for h := 0; h < lay.hid; h++ {
				i := b*lay.hid + h
				s := core.AsFloat64(ihPost.Data[i]) + core.AsFloat64(hhPost.Data[i])
				idx := b*lay.seq*lay.hid + t*lay.hid + h
				pre.Data[idx] = core.FromFloat64[T](s)
				th := math.Tanh(s)
				post.Data[idx] = core.FromFloat64[T](th)
				hPrev[i] = core.FromFloat64[T](th)
			}
		}
	}
	return pre, post, nil
}

func backwardViaDense[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	lay, err := parseLayout(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	if gradOut == nil || pre == nil {
		return nil, nil, fmt.Errorf("rnn: nil gradOut/pre")
	}
	want := lay.batch * lay.seq * lay.hid
	if gradOut.Len() < want || pre.Len() < want {
		return nil, nil, fmt.Errorf("rnn: grad/pre short")
	}

	ihBe, hhBe := l.IH.Exec.Backend, l.HH.Exec.Backend
	if l.Exec.Backend == core.BackendWebGPU {
		l.IH.Exec.Backend = core.BackendCPUTiled
		l.HH.Exec.Backend = core.BackendCPUTiled
	}
	defer func() {
		l.IH.Exec.Backend = ihBe
		l.HH.Exec.Backend = hhBe
	}()

	ihN := lay.hid * lay.in
	hhN := lay.hid * lay.hid
	dIH := make([]float64, ihN)
	dHH := make([]float64, hhN)
	dB := make([]float64, lay.hid)
	gi := make([]float64, lay.batch*lay.seq*lay.in)
	gH := make([]float64, lay.batch*lay.hid) // zeros

	// Rebuild hidden states from pre (h_t = tanh(pre_t)); h_{-1}=0
	hSeq := make([]T, lay.batch*lay.seq*lay.hid)
	for i := 0; i < want; i++ {
		hSeq[i] = core.FromFloat64[T](math.Tanh(core.AsFloat64(pre.Data[i])))
	}

	for t := lay.seq - 1; t >= 0; t-- {
		gPreT := core.NewTensor[T](lay.batch, lay.hid)
		for b := 0; b < lay.batch; b++ {
			for h := 0; h < lay.hid; h++ {
				idx := b*lay.seq*lay.hid + t*lay.hid + h
				hVal := math.Tanh(core.AsFloat64(pre.Data[idx]))
				g := (gH[b*lay.hid+h] + core.AsFloat64(gradOut.Data[idx])) * (1 - hVal*hVal)
				gPreT.Data[b*lay.hid+h] = core.FromFloat64[T](g)
				dB[h] += g
			}
		}

		xt := xAt(input, lay, t)
		var hPrev *core.Tensor[T]
		if t == 0 {
			hPrev = core.NewTensor[T](lay.batch, lay.hid) // zeros
		} else {
			hPrev = core.NewTensor[T](lay.batch, lay.hid)
			for b := 0; b < lay.batch; b++ {
				copy(hPrev.Data[b*lay.hid:(b+1)*lay.hid],
					hSeq[b*lay.seq*lay.hid+(t-1)*lay.hid:b*lay.seq*lay.hid+t*lay.hid])
			}
		}

		ihPre, _, err := dense.Forward(l.IH, xt)
		if err != nil {
			return nil, nil, fmt.Errorf("rnn IH recompute t=%d: %w", t, err)
		}
		hhPre, _, err := dense.Forward(l.HH, hPrev)
		if err != nil {
			return nil, nil, fmt.Errorf("rnn HH recompute t=%d: %w", t, err)
		}

		gx, dWIH, err := dense.Backward(l.IH, gPreT, xt, ihPre)
		if err != nil {
			return nil, nil, fmt.Errorf("rnn IH bwd t=%d: %w", t, err)
		}
		gh, dWHH, err := dense.Backward(l.HH, gPreT, hPrev, hhPre)
		if err != nil {
			return nil, nil, fmt.Errorf("rnn HH bwd t=%d: %w", t, err)
		}

		for i := 0; i < ihN; i++ {
			dIH[i] += core.AsFloat64(dWIH.Data[i])
		}
		for i := 0; i < hhN; i++ {
			dHH[i] += core.AsFloat64(dWHH.Data[i])
		}
		for b := 0; b < lay.batch; b++ {
			for i := 0; i < lay.in; i++ {
				gi[b*lay.seq*lay.in+t*lay.in+i] += core.AsFloat64(gx.Data[b*lay.in+i])
			}
			for h := 0; h < lay.hid; h++ {
				gH[b*lay.hid+h] = core.AsFloat64(gh.Data[b*lay.hid+h])
			}
		}
	}

	gradIn = core.NewTensor[T](lay.batch, lay.seq, lay.in)
	for i := range gradIn.Data {
		gradIn.Data[i] = core.FromFloat64[T](gi[i])
	}
	gradW = core.NewTensor[T](1, l.GradWSize())
	off := 0
	for i := 0; i < ihN; i++ {
		gradW.Data[off+i] = core.FromFloat64[T](dIH[i])
	}
	off += ihN
	for i := 0; i < hhN; i++ {
		gradW.Data[off+i] = core.FromFloat64[T](dHH[i])
	}
	off += hhN
	for i := 0; i < lay.hid; i++ {
		gradW.Data[off+i] = core.FromFloat64[T](dB[i])
	}
	return gradIn, gradW, nil
}
