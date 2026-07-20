package gdn

import (
	"fmt"
	"math"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
)

// gdnStep captures the intermediates of one ForwardDecode token needed for
// backwardStep. State/ConvState are recurrent and treated as stop-gradient
// across tokens (see doc.go) — only the current-token contribution to those
// buffers is stored.
type gdnStep struct {
	x                  []float32 // H
	qkv                []float32 // convDim (raw InQKV projection)
	convHist           []float32 // convDim*(k-1), conv history BEFORE this token
	acc                []float32 // convDim, pre-silu conv accumulator
	mixed              []float32 // convDim, post-silu (q_raw|key_raw|v)
	qInv, kInv         []float32 // nk, 1/||.|| per key-head group
	qRep, kRep         []float32 // nh*hdK, post l2norm+broadcast(+scale for q)
	betaRaw, aRaw      []float32 // nh
	beta, g            []float32 // nh, sigmoid(betaRaw) / gate value
	stateOld           []float32 // nh*hdK*hdV, State BEFORE decay this token
	stateAfterDecay    []float32 // nh*hdK*hdV, State*gt BEFORE delta injection
	kvMem              []float32 // valDim (nh*hdV), stateAfterDecay^T @ k per head
	delta              []float32 // valDim (nh*hdV), delta-rule update per head
	corePre            []float32 // valDim, pre-rmsNormGated head outputs
	z                  []float32 // valDim, InZ projection at this token
	coreNormed         []float32 // valDim, post rmsNormGated (Out matvec input)
}

// gdnGrads accumulates weight gradients across all tokens/batch items in one Backward call.
type gdnGrads struct {
	cfg                                      Config
	dInQKV, dInZ, dInB, dInA, dOut           []float32
	dConvWeight, dALog, dDtBias, dNormGamma  []float32
}

func newGDNGrads(cfg Config) *gdnGrads {
	cd, vd, nh, h := cfg.convDim(), cfg.valueDim(), cfg.NumValueHeads, cfg.HiddenSize
	k := cfg.ConvKernel
	if k < 1 {
		k = 1
	}
	return &gdnGrads{
		cfg:         cfg,
		dInQKV:      make([]float32, cd*h),
		dInZ:        make([]float32, vd*h),
		dInB:        make([]float32, nh*h),
		dInA:        make([]float32, nh*h),
		dOut:        make([]float32, h*vd),
		dConvWeight: make([]float32, cd*k),
		dALog:       make([]float32, nh),
		dDtBias:     make([]float32, nh),
		dNormGamma:  make([]float32, cfg.ValueHeadDim),
	}
}

func flattenGDNGrads[T core.Numeric](g *gdnGrads) *core.Tensor[T] {
	n := len(g.dInQKV) + len(g.dInZ) + len(g.dInB) + len(g.dInA) + len(g.dOut) +
		len(g.dConvWeight) + len(g.dALog) + len(g.dDtBias) + len(g.dNormGamma)
	out := core.NewTensor[T](n)
	off := 0
	put := func(s []float32) {
		for _, v := range s {
			out.Data[off] = core.FromFloat64[T](float64(v))
			off++
		}
	}
	put(g.dInQKV)
	put(g.dInZ)
	put(g.dInB)
	put(g.dInA)
	put(g.dOut)
	put(g.dConvWeight)
	put(g.dALog)
	put(g.dDtBias)
	put(g.dNormGamma)
	return out
}

// GradWSize is the concatenated InQKV+InZ+InB+InA+Out+ConvWeight+ALog+DtBias+NormGamma length.
func (l *Layer) GradWSize() int {
	if l == nil {
		return 0
	}
	c := l.Cfg
	cd, vd, nh, h := c.convDim(), c.valueDim(), c.NumValueHeads, c.HiddenSize
	k := c.ConvKernel
	if k < 1 {
		k = 1
	}
	return cd*h + vd*h + nh*h + nh*h + h*vd + cd*k + nh + nh + c.ValueHeadDim
}

// forwardDecodeTape runs one token through GDN (mirrors ForwardDecode) while recording
// every intermediate backwardStep needs. Mutates l.ConvState/l.State exactly like
// ForwardDecode so subsequent tokens see identical recurrent state.
func (l *Layer) forwardDecodeTape(x, y []float32) (*gdnStep, error) {
	if l == nil {
		return nil, fmt.Errorf("gdn: nil layer")
	}
	c := l.Cfg
	h := c.HiddenSize
	if len(x) < h || len(y) < h {
		return nil, fmt.Errorf("gdn: shape")
	}
	l.ensureScratch()
	keyDim, valDim, convDim := c.keyDim(), c.valueDim(), c.convDim()
	nh, nk := c.NumValueHeads, c.NumKeyHeads
	rep := nh / nk
	hdK, hdV := c.KeyHeadDim, c.ValueHeadDim
	k := c.ConvKernel
	if k < 1 {
		k = 1
	}

	qkv := make([]float32, convDim)
	if err := matVec(l.InQKV, x, qkv, l.UseGPU); err != nil {
		return nil, err
	}
	z := make([]float32, valDim)
	if err := matVec(l.InZ, x, z, l.UseGPU); err != nil {
		return nil, err
	}
	betaRaw := make([]float32, nh)
	if err := matVec(l.InB, x, betaRaw, l.UseGPU); err != nil {
		return nil, err
	}
	aRaw := make([]float32, nh)
	if err := matVec(l.InA, x, aRaw, l.UseGPU); err != nil {
		return nil, err
	}

	convHist := append([]float32(nil), l.ConvState...)
	acc := make([]float32, convDim)
	mixed := make([]float32, convDim)
	for ch := 0; ch < convDim; ch++ {
		var a float32
		for t := 0; t < k-1; t++ {
			a += l.ConvWeight[ch*k+t] * l.ConvState[ch*(k-1)+t]
		}
		a += l.ConvWeight[ch*k+(k-1)] * qkv[ch]
		acc[ch] = a
		mixed[ch] = silu(a)
	}
	for ch := 0; ch < convDim; ch++ {
		base := ch * (k - 1)
		if k > 2 {
			copy(l.ConvState[base:base+k-2], l.ConvState[base+1:base+k-1])
		}
		if k > 1 {
			l.ConvState[base+k-2] = qkv[ch]
		}
	}

	q := mixed[:keyDim]
	key := mixed[keyDim : keyDim*2]
	v := mixed[keyDim*2 : keyDim*2+valDim]

	qRep := make([]float32, nh*hdK)
	kRep := make([]float32, nh*hdK)
	qInv := make([]float32, nk)
	kInv := make([]float32, nk)
	scale := float32(1 / math.Sqrt(float64(hdK)))
	for hi := 0; hi < nk; hi++ {
		qs := q[hi*hdK : (hi+1)*hdK]
		ks := key[hi*hdK : (hi+1)*hdK]
		var sq, sk float64
		for _, vv := range qs {
			sq += float64(vv) * float64(vv)
		}
		for _, vv := range ks {
			sk += float64(vv) * float64(vv)
		}
		invq := float32(1 / math.Sqrt(sq+1e-6))
		invk := float32(1 / math.Sqrt(sk+1e-6))
		qInv[hi] = invq
		kInv[hi] = invk
		for r := 0; r < rep; r++ {
			dst := (hi*rep + r) * hdK
			for j := 0; j < hdK; j++ {
				qRep[dst+j] = qs[j] * invq * scale
				kRep[dst+j] = ks[j] * invk
			}
		}
	}

	beta := make([]float32, nh)
	gGate := make([]float32, nh)
	for i := 0; i < nh; i++ {
		beta[i] = 1 / (1 + float32(math.Exp(float64(-betaRaw[i]))))
		al := float64(l.ALog[i])
		dt := float64(aRaw[i] + l.DtBias[i])
		gGate[i] = float32(-math.Exp(al) * softplus(dt))
	}

	if !l.HasState {
		for i := range l.State {
			l.State[i] = 0
		}
		l.HasState = true
	}
	stateOld := append([]float32(nil), l.State...)
	state := append([]float32(nil), l.State...)
	stateAfterDecay := make([]float32, len(state))
	kvMem := make([]float32, valDim)
	delta := make([]float32, valDim)
	corePre := make([]float32, valDim)

	for hIdx := 0; hIdx < nh; hIdx++ {
		st := state[hIdx*hdK*hdV : (hIdx+1)*hdK*hdV]
		gt := float32(math.Exp(float64(gGate[hIdx])))
		bt := beta[hIdx]
		qt := qRep[hIdx*hdK : (hIdx+1)*hdK]
		kt := kRep[hIdx*hdK : (hIdx+1)*hdK]
		vt := v[hIdx*hdV : (hIdx+1)*hdV]

		for i := range st {
			st[i] *= gt
		}
		copy(stateAfterDecay[hIdx*hdK*hdV:(hIdx+1)*hdK*hdV], st)

		kvm := kvMem[hIdx*hdV : (hIdx+1)*hdV]
		for d := 0; d < hdV; d++ {
			var s float32
			for j := 0; j < hdK; j++ {
				s += st[j*hdV+d] * kt[j]
			}
			kvm[d] = s
		}
		dl := delta[hIdx*hdV : (hIdx+1)*hdV]
		for d := 0; d < hdV; d++ {
			dl[d] = (vt[d] - kvm[d]) * bt
		}
		for j := 0; j < hdK; j++ {
			for d := 0; d < hdV; d++ {
				st[j*hdV+d] += kt[j] * dl[d]
			}
		}
		outH := corePre[hIdx*hdV : (hIdx+1)*hdV]
		for d := 0; d < hdV; d++ {
			var s float32
			for j := 0; j < hdK; j++ {
				s += st[j*hdV+d] * qt[j]
			}
			outH[d] = s
		}
	}
	copy(l.State, state)

	coreNormed := append([]float32(nil), corePre...)
	for hIdx := 0; hIdx < nh; hIdx++ {
		o := coreNormed[hIdx*hdV : (hIdx+1)*hdV]
		zz := z[hIdx*hdV : (hIdx+1)*hdV]
		rmsNormGated(o, zz, l.NormGamma, float32(c.Eps))
	}

	if err := matVec(l.Out, coreNormed[:valDim], y, l.UseGPU); err != nil {
		return nil, err
	}

	return &gdnStep{
		x:               append([]float32(nil), x[:h]...),
		qkv:             qkv,
		convHist:        convHist,
		acc:             acc,
		mixed:           mixed,
		qInv:            qInv,
		kInv:            kInv,
		qRep:            qRep,
		kRep:            kRep,
		betaRaw:         betaRaw,
		aRaw:            aRaw,
		beta:            beta,
		g:               gGate,
		stateOld:        stateOld,
		stateAfterDecay: stateAfterDecay,
		kvMem:           kvMem,
		delta:           delta,
		corePre:         corePre,
		z:               z,
		coreNormed:      coreNormed,
	}, nil
}

// l2normBackward returns dx for y = x/||x||_2 (eps inside the norm), given dy and the
// forward inputs xRaw and inv (=1/||xRaw||).
func l2normBackward(xRaw []float32, inv float32, dy []float32) []float32 {
	n := len(xRaw)
	dx := make([]float32, n)
	var s float64
	for i := 0; i < n; i++ {
		y := xRaw[i] * inv
		s += float64(dy[i]) * float64(y)
	}
	sf := float32(s)
	for j := 0; j < n; j++ {
		y := xRaw[j] * inv
		dx[j] = inv * (dy[j] - y*sf)
	}
	return dx
}

// accumOuter adds gy⊗x into dW (row-major, rows×cols): dW[r,c] += gy[r]*x[c].
func accumOuter(dW, gy, x []float32, rows, cols int) {
	for r := 0; r < rows; r++ {
		gr := gy[r]
		if gr == 0 {
			continue
		}
		base := r * cols
		for c := 0; c < cols; c++ {
			dW[base+c] += gr * x[c]
		}
	}
}

// backwardStep consumes dy (grad wrt this token's output, len H) and this token's tape,
// accumulates weight grads into g, and returns dx (grad wrt this token's input, len H).
//
// Exact: Out, NormGamma, InZ, and the direct current-token InQKV/InB/InA contribution.
// Approximate (stop-gradient): the recurrent State and ConvState history — gradient does
// not flow across tokens through those buffers (see doc.go).
func (l *Layer) backwardStep(tp *gdnStep, dy []float32, g *gdnGrads) ([]float32, error) {
	c := l.Cfg
	h := c.HiddenSize
	keyDim, valDim, convDim := c.keyDim(), c.valueDim(), c.convDim()
	nh, nk := c.NumValueHeads, c.NumKeyHeads
	rep := nh / nk
	hdK, hdV := c.KeyHeadDim, c.ValueHeadDim
	k := c.ConvKernel
	if k < 1 {
		k = 1
	}

	// Out: y = Out @ coreNormed.
	dCoreNormed := make([]float32, valDim)
	if err := quant.MatVecT(l.Out, dy, dCoreNormed); err != nil {
		return nil, fmt.Errorf("gdn bwd Out: %w", err)
	}
	accumOuter(g.dOut, dy, tp.coreNormed[:valDim], h, valDim)

	// rmsNormGated backward per head (exact).
	dCorePre := make([]float32, valDim)
	dZ := make([]float32, valDim)
	for hIdx := 0; hIdx < nh; hIdx++ {
		xr := tp.corePre[hIdx*hdV : (hIdx+1)*hdV]
		zz := tp.z[hIdx*hdV : (hIdx+1)*hdV]
		dyh := dCoreNormed[hIdx*hdV : (hIdx+1)*hdV]
		n := len(xr)
		var mean float32
		for _, v := range xr {
			mean += v * v
		}
		mean /= float32(n)
		inv := 1 / float32(math.Sqrt(float64(mean+float32(c.Eps))))
		var s float64
		for i := 0; i < n; i++ {
			gamma := float32(1)
			if i < len(l.NormGamma) {
				gamma = l.NormGamma[i]
			}
			sz := silu(zz[i])
			ci := gamma * sz
			s += float64(dyh[i]) * float64(ci) * float64(xr[i])
		}
		sf := float32(s)
		for i := 0; i < n; i++ {
			gamma := float32(1)
			if i < len(l.NormGamma) {
				gamma = l.NormGamma[i]
			}
			sz := silu(zz[i])
			ci := gamma * sz
			dCorePre[hIdx*hdV+i] = inv*ci*dyh[i] - inv*inv*inv/float32(n)*xr[i]*sf
			dZ[hIdx*hdV+i] = dyh[i] * xr[i] * inv * gamma * siluDeriv(zz[i])
			if i < len(g.dNormGamma) {
				g.dNormGamma[i] += dyh[i] * xr[i] * inv * sz
			}
		}
	}

	// Recurrent core (delta rule), state treated as stop-gradient across tokens.
	dMixed := make([]float32, convDim)
	dBetaRaw := make([]float32, nh)
	dARaw := make([]float32, nh)
	dQRep := make([]float32, nh*hdK)
	dKRep := make([]float32, nh*hdK)
	for hIdx := 0; hIdx < nh; hIdx++ {
		doutH := dCorePre[hIdx*hdV : (hIdx+1)*hdV]
		stAfter := tp.stateAfterDecay[hIdx*hdK*hdV : (hIdx+1)*hdK*hdV]
		stOld := tp.stateOld[hIdx*hdK*hdV : (hIdx+1)*hdK*hdV]
		dl := tp.delta[hIdx*hdV : (hIdx+1)*hdV]
		qt := tp.qRep[hIdx*hdK : (hIdx+1)*hdK]
		kt := tp.kRep[hIdx*hdK : (hIdx+1)*hdK]
		v := tp.mixed[keyDim*2+hIdx*hdV : keyDim*2+(hIdx+1)*hdV]

		dStateFinal := make([]float32, hdK*hdV)
		dqt := make([]float32, hdK)
		for j := 0; j < hdK; j++ {
			var s float32
			for d := 0; d < hdV; d++ {
				stFinal := stAfter[j*hdV+d] + kt[j]*dl[d]
				dStateFinal[j*hdV+d] = doutH[d] * qt[j]
				s += stFinal * doutH[d]
			}
			dqt[j] = s
		}
		dkt := make([]float32, hdK)
		ddelta := make([]float32, hdV)
		dStateAfter := make([]float32, hdK*hdV)
		for j := 0; j < hdK; j++ {
			var dktj float32
			for d := 0; d < hdV; d++ {
				dsf := dStateFinal[j*hdV+d]
				dStateAfter[j*hdV+d] += dsf
				dktj += dl[d] * dsf
				ddelta[d] += kt[j] * dsf
			}
			dkt[j] += dktj
		}
		bt := tp.beta[hIdx]
		kvm := tp.kvMem[hIdx*hdV : (hIdx+1)*hdV]
		var dbt float32
		dKvMem := make([]float32, hdV)
		for d := 0; d < hdV; d++ {
			ddl := ddelta[d]
			// delta[d] = (v[d]-kvMem[d])*bt
			dbt += ddl * (v[d] - kvm[d])
			dKvMem[d] = -ddl * bt
			dMixed[keyDim*2+hIdx*hdV+d] += ddl * bt // dv
		}
		for j := 0; j < hdK; j++ {
			var dktj float32
			for d := 0; d < hdV; d++ {
				dStateAfter[j*hdV+d] += dKvMem[d] * kt[j]
				dktj += dKvMem[d] * stAfter[j*hdV+d]
			}
			dkt[j] += dktj
		}

		gt := float32(math.Exp(float64(tp.g[hIdx])))
		var dgt float32
		for j := 0; j < hdK; j++ {
			for d := 0; d < hdV; d++ {
				dgt += dStateAfter[j*hdV+d] * stOld[j*hdV+d]
			}
		}
		dGate := dgt * gt
		al := float64(l.ALog[hIdx])
		dtRaw := float64(tp.aRaw[hIdx] + l.DtBias[hIdx])
		sig := float32(1 / (1 + math.Exp(-dtRaw)))
		dDtRaw := dGate * float32(-math.Exp(al)) * sig
		dARaw[hIdx] += dDtRaw
		g.dDtBias[hIdx] += dDtRaw
		g.dALog[hIdx] += dGate * tp.g[hIdx]

		dBetaRaw[hIdx] += dbt * bt * (1 - bt)

		for j := 0; j < hdK; j++ {
			dQRep[hIdx*hdK+j] = dqt[j]
			dKRep[hIdx*hdK+j] = dkt[j]
		}
	}

	// Route dQRep/dKRep back through broadcast+scale into per-key-head l2norm backward.
	scale := float32(1 / math.Sqrt(float64(hdK)))
	dq := make([]float32, keyDim)
	dk := make([]float32, keyDim)
	for hi := 0; hi < nk; hi++ {
		dqNorm := make([]float32, hdK)
		dkNorm := make([]float32, hdK)
		for r := 0; r < rep; r++ {
			hIdx := hi*rep + r
			for j := 0; j < hdK; j++ {
				dqNorm[j] += dQRep[hIdx*hdK+j] * scale
				dkNorm[j] += dKRep[hIdx*hdK+j]
			}
		}
		qRaw := tp.mixed[hi*hdK : (hi+1)*hdK]
		kRaw := tp.mixed[keyDim+hi*hdK : keyDim+(hi+1)*hdK]
		dqRaw := l2normBackward(qRaw, tp.qInv[hi], dqNorm)
		dkRaw := l2normBackward(kRaw, tp.kInv[hi], dkNorm)
		copy(dq[hi*hdK:(hi+1)*hdK], dqRaw)
		copy(dk[hi*hdK:(hi+1)*hdK], dkRaw)
	}
	copy(dMixed[0:keyDim], dq)
	for i := 0; i < keyDim; i++ {
		dMixed[keyDim+i] += dk[i]
	}

	// Conv + silu backward (current-token contribution only; conv history is stop-gradient).
	dQKV := make([]float32, convDim)
	kk := k
	for ch := 0; ch < convDim; ch++ {
		dAcc := dMixed[ch] * siluDeriv(tp.acc[ch])
		for t := 0; t < kk-1; t++ {
			g.dConvWeight[ch*kk+t] += dAcc * tp.convHist[ch*(kk-1)+t]
		}
		g.dConvWeight[ch*kk+(kk-1)] += dAcc * tp.qkv[ch]
		dQKV[ch] += dAcc * l.ConvWeight[ch*kk+(kk-1)]
	}

	// Linear maps: MatVecT for dx, outer product for weight grads.
	dx := make([]float32, h)
	if err := quant.MatVecT(l.InQKV, dQKV, dx); err != nil {
		return nil, fmt.Errorf("gdn bwd InQKV: %w", err)
	}
	accumOuter(g.dInQKV, dQKV, tp.x[:h], convDim, h)

	if err := quant.MatVecT(l.InZ, dZ, dx); err != nil {
		return nil, fmt.Errorf("gdn bwd InZ: %w", err)
	}
	accumOuter(g.dInZ, dZ, tp.x[:h], valDim, h)

	if err := quant.MatVecT(l.InB, dBetaRaw, dx); err != nil {
		return nil, fmt.Errorf("gdn bwd InB: %w", err)
	}
	accumOuter(g.dInB, dBetaRaw, tp.x[:h], nh, h)

	if err := quant.MatVecT(l.InA, dARaw, dx); err != nil {
		return nil, fmt.Errorf("gdn bwd InA: %w", err)
	}
	accumOuter(g.dInA, dARaw, tp.x[:h], nh, h)

	return dx, nil
}

func siluDeriv(x float32) float32 {
	s := 1 / (1 + float32(math.Exp(float64(-x))))
	return s + x*s*(1-s)
}

// Backward runs truncated-BPTT (per-token exact where noted; recurrent state is
// stop-gradient across tokens — see doc.go) over [B,T,H] gradOut/input.
func Backward[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if l == nil || input == nil || gradOut == nil {
		return nil, nil, fmt.Errorf("gdn: Backward nil")
	}
	if len(input.Shape) != 3 || input.Shape[2] != l.Cfg.HiddenSize {
		return nil, nil, fmt.Errorf("gdn: need [B,T,%d], got %v", l.Cfg.HiddenSize, input.Shape)
	}
	b, t, h := input.Shape[0], input.Shape[1], input.Shape[2]
	if len(gradOut.Shape) != 3 || gradOut.Shape[0] != b || gradOut.Shape[1] != t || gradOut.Shape[2] != h {
		return nil, nil, fmt.Errorf("gdn: gradOut shape %v != input %v", gradOut.Shape, input.Shape)
	}
	grads := newGDNGrads(l.Cfg)
	gradIn = core.NewTensor[T](b, t, h)

	for bi := 0; bi < b; bi++ {
		l.Reset()
		steps := make([]*gdnStep, t)
		for ti := 0; ti < t; ti++ {
			x := make([]float32, h)
			base := (bi*t + ti) * h
			for i := 0; i < h; i++ {
				x[i] = float32(core.AsFloat64(input.Data[base+i]))
			}
			y := make([]float32, h)
			st, err := l.forwardDecodeTape(x, y)
			if err != nil {
				return nil, nil, fmt.Errorf("gdn bwd fwd-tape t=%d: %w", ti, err)
			}
			steps[ti] = st
		}
		for ti := t - 1; ti >= 0; ti-- {
			dy := make([]float32, h)
			base := (bi*t + ti) * h
			for i := 0; i < h; i++ {
				dy[i] = float32(core.AsFloat64(gradOut.Data[base+i]))
			}
			dx, err := l.backwardStep(steps[ti], dy, grads)
			if err != nil {
				return nil, nil, fmt.Errorf("gdn bwd step t=%d: %w", ti, err)
			}
			for i := 0; i < h; i++ {
				gradIn.Data[base+i] = core.FromFloat64[T](float64(dx[i]))
			}
		}
	}
	gradW = flattenGDNGrads[T](grads)
	return gradIn, gradW, nil
}

// applyBlobSGD unpacks *bp, applies w -= lr*dW elementwise, and re-Packs FormatNone.
func applyBlobSGD(bp **quant.Blob, dW []float32, lr float64) error {
	b := *bp
	if b == nil {
		return fmt.Errorf("gdn: nil blob")
	}
	f32, err := quant.Unpack(b)
	if err != nil {
		return fmt.Errorf("gdn: unpack: %w", err)
	}
	n := b.Rows * b.Cols
	if len(f32) < n || len(dW) < n {
		return fmt.Errorf("gdn: applyBlobSGD shape")
	}
	for i := 0; i < n; i++ {
		f32[i] -= float32(lr) * dW[i]
	}
	nb, err := quant.Pack(quant.FormatNone, f32[:n], b.Rows, b.Cols)
	if err != nil {
		return fmt.Errorf("gdn: pack: %w", err)
	}
	*bp = nb
	return nil
}

// ApplyGradSGD unpacks each projection blob to f32, applies SGD, and re-Packs FormatNone.
// ConvWeight/ALog/DtBias/NormGamma (plain []float32) update directly.
func ApplyGradSGD[T core.Numeric](l *Layer, dW *core.Tensor[T], lr float64) error {
	if l == nil || dW == nil {
		return fmt.Errorf("gdn: ApplyGradSGD nil")
	}
	need := l.GradWSize()
	if dW.Len() < need {
		return fmt.Errorf("gdn: dW len %d < %d", dW.Len(), need)
	}
	f32 := make([]float32, need)
	for i := 0; i < need; i++ {
		f32[i] = float32(core.AsFloat64(dW.Data[i]))
	}
	off := 0
	take := func(n int) []float32 {
		s := f32[off : off+n]
		off += n
		return s
	}
	c := l.Cfg
	cd, vd, nh, h := c.convDim(), c.valueDim(), c.NumValueHeads, c.HiddenSize
	k := c.ConvKernel
	if k < 1 {
		k = 1
	}

	if err := applyBlobSGD(&l.InQKV, take(cd*h), lr); err != nil {
		return err
	}
	if err := applyBlobSGD(&l.InZ, take(vd*h), lr); err != nil {
		return err
	}
	if err := applyBlobSGD(&l.InB, take(nh*h), lr); err != nil {
		return err
	}
	if err := applyBlobSGD(&l.InA, take(nh*h), lr); err != nil {
		return err
	}
	if err := applyBlobSGD(&l.Out, take(h*vd), lr); err != nil {
		return err
	}
	dConv := take(cd * k)
	for i := range l.ConvWeight {
		l.ConvWeight[i] -= float32(lr) * dConv[i]
	}
	dALog := take(nh)
	for i := range l.ALog {
		l.ALog[i] -= float32(lr) * dALog[i]
	}
	dDtBias := take(nh)
	for i := range l.DtBias {
		l.DtBias[i] -= float32(lr) * dDtBias[i]
	}
	dGamma := take(c.ValueHeadDim)
	for i := range l.NormGamma {
		l.NormGamma[i] -= float32(lr) * dGamma[i]
	}
	return nil
}
