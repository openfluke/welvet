package gdn

import (
	"fmt"
	"math"

	"github.com/openfluke/welvet/dense"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/weights"
)

// Config holds Gated DeltaNet shape (Qwen3.5 / Bonsai linear_attention layers).
type Config struct {
	HiddenSize    int
	NumKeyHeads   int
	NumValueHeads int
	KeyHeadDim    int
	ValueHeadDim  int
	ConvKernel    int
	Eps           float64
}

func (c Config) keyDim() int   { return c.NumKeyHeads * c.KeyHeadDim }
func (c Config) valueDim() int { return c.NumValueHeads * c.ValueHeadDim }
func (c Config) convDim() int  { return c.keyDim()*2 + c.valueDim() }

// Layer is one Gated DeltaNet mixer with binary-packed projections.
type Layer struct {
	Cfg Config

	InQKV, InZ, InB, InA, Out *quant.Blob
	ConvWeight                []float32 // [conv_dim * kernel] row-major ch,k
	ALog, DtBias              []float32 // [num_v_heads]
	NormGamma                 []float32 // [value_head_dim], already (1+w)
	UseGPU                    bool      // WebGPU BinaryG128 GEMV for projections

	ConvState []float32
	State     []float32
	HasState  bool

	// persistent scratch (no aliasing across live tensors)
	qkv, z, betaRaw, aRaw []float32
	mixed, qRep, kRep     []float32
	beta, g, core         []float32
	kvMem, delta          []float32
}

// Reset clears conv + recurrent state.
func (l *Layer) Reset() {
	if l == nil {
		return
	}
	l.HasState = false
	for i := range l.ConvState {
		l.ConvState[i] = 0
	}
	for i := range l.State {
		l.State[i] = 0
	}
}

func (l *Layer) ensureScratch() {
	c := l.Cfg
	cd, vd, nh := c.convDim(), c.valueDim(), c.NumValueHeads
	hdK, hdV := c.KeyHeadDim, c.ValueHeadDim
	l.qkv = grow(l.qkv, cd)
	l.z = grow(l.z, vd)
	l.betaRaw = grow(l.betaRaw, nh)
	l.aRaw = grow(l.aRaw, nh)
	l.mixed = grow(l.mixed, cd)
	l.qRep = grow(l.qRep, nh*hdK)
	l.kRep = grow(l.kRep, nh*hdK)
	l.beta = grow(l.beta, nh)
	l.g = grow(l.g, nh)
	l.core = grow(l.core, nh*hdV)
	l.kvMem = grow(l.kvMem, hdV)
	l.delta = grow(l.delta, hdV)
	k := c.ConvKernel
	if k < 1 {
		k = 1
	}
	l.ConvState = grow(l.ConvState, cd*(k-1))
	l.State = grow(l.State, nh*hdK*hdV)
}

func grow(s []float32, n int) []float32 {
	if cap(s) >= n {
		return s[:n]
	}
	return make([]float32, n)
}

// ForwardDecode runs one token through GDN (seq_len=1 recurrent path).
func (l *Layer) ForwardDecode(x, y []float32) error {
	if l == nil {
		return fmt.Errorf("gdn: nil layer")
	}
	c := l.Cfg
	h := c.HiddenSize
	if len(x) < h || len(y) < h {
		return fmt.Errorf("gdn: shape")
	}
	l.ensureScratch()
	keyDim, valDim := c.keyDim(), c.valueDim()
	convDim := c.convDim()

	if err := matVec(l.InQKV, x, l.qkv, l.UseGPU); err != nil {
		return err
	}
	if err := matVec(l.InZ, x, l.z, l.UseGPU); err != nil {
		return err
	}
	if err := matVec(l.InB, x, l.betaRaw, l.UseGPU); err != nil {
		return err
	}
	if err := matVec(l.InA, x, l.aRaw, l.UseGPU); err != nil {
		return err
	}

	k := c.ConvKernel
	if k < 1 {
		k = 1
	}
	for ch := 0; ch < convDim; ch++ {
		var acc float32
		for t := 0; t < k-1; t++ {
			acc += l.ConvWeight[ch*k+t] * l.ConvState[ch*(k-1)+t]
		}
		acc += l.ConvWeight[ch*k+(k-1)] * l.qkv[ch]
		l.mixed[ch] = silu(acc)
	}
	for ch := 0; ch < convDim; ch++ {
		base := ch * (k - 1)
		if k > 2 {
			copy(l.ConvState[base:base+k-2], l.ConvState[base+1:base+k-1])
		}
		if k > 1 {
			l.ConvState[base+k-2] = l.qkv[ch]
		}
	}

	q := l.mixed[:keyDim]
	key := l.mixed[keyDim : keyDim*2]
	v := l.mixed[keyDim*2 : keyDim*2+valDim]

	nh, nk := c.NumValueHeads, c.NumKeyHeads
	rep := nh / nk
	hdK, hdV := c.KeyHeadDim, c.ValueHeadDim

	for hi := 0; hi < nk; hi++ {
		l2normInPlace(q[hi*hdK:(hi+1)*hdK], 1e-6)
		l2normInPlace(key[hi*hdK:(hi+1)*hdK], 1e-6)
		for r := 0; r < rep; r++ {
			dst := (hi*rep + r) * hdK
			copy(l.qRep[dst:dst+hdK], q[hi*hdK:(hi+1)*hdK])
			copy(l.kRep[dst:dst+hdK], key[hi*hdK:(hi+1)*hdK])
		}
	}
	scale := float32(1 / math.Sqrt(float64(hdK)))
	for i := range l.qRep {
		l.qRep[i] *= scale
	}

	for i := 0; i < nh; i++ {
		l.beta[i] = 1 / (1 + float32(math.Exp(float64(-l.betaRaw[i]))))
		al := float64(l.ALog[i])
		dt := float64(l.aRaw[i] + l.DtBias[i])
		l.g[i] = float32(-math.Exp(al) * softplus(dt))
	}

	if !l.HasState {
		for i := range l.State {
			l.State[i] = 0
		}
		l.HasState = true
	}

	for hIdx := 0; hIdx < nh; hIdx++ {
		st := l.State[hIdx*hdK*hdV : (hIdx+1)*hdK*hdV]
		gt := float32(math.Exp(float64(l.g[hIdx])))
		bt := l.beta[hIdx]
		qt := l.qRep[hIdx*hdK : (hIdx+1)*hdK]
		kt := l.kRep[hIdx*hdK : (hIdx+1)*hdK]
		vt := v[hIdx*hdV : (hIdx+1)*hdV]

		for i := range st {
			st[i] *= gt
		}
		for d := 0; d < hdV; d++ {
			var s float32
			for j := 0; j < hdK; j++ {
				s += st[j*hdV+d] * kt[j]
			}
			l.kvMem[d] = s
		}
		for d := 0; d < hdV; d++ {
			l.delta[d] = (vt[d] - l.kvMem[d]) * bt
		}
		for j := 0; j < hdK; j++ {
			for d := 0; d < hdV; d++ {
				st[j*hdV+d] += kt[j] * l.delta[d]
			}
		}
		outH := l.core[hIdx*hdV : (hIdx+1)*hdV]
		for d := 0; d < hdV; d++ {
			var s float32
			for j := 0; j < hdK; j++ {
				s += st[j*hdV+d] * qt[j]
			}
			outH[d] = s
		}
	}

	for hIdx := 0; hIdx < nh; hIdx++ {
		o := l.core[hIdx*hdV : (hIdx+1)*hdV]
		zz := l.z[hIdx*hdV : (hIdx+1)*hdV]
		rmsNormGated(o, zz, l.NormGamma, float32(c.Eps))
	}

	return matVec(l.Out, l.core[:valDim], y, l.UseGPU)
}

func matVec(b *quant.Blob, x, y []float32, useGPU bool) error {
	if b == nil {
		return fmt.Errorf("gdn: nil blob")
	}
	if useGPU {
		ws, err := weights.FromBlob(b)
		if err != nil {
			return err
		}
		return dense.MatVecWebGPU(ws, x, y, 1, b.Cols, b.Rows)
	}
	return quant.MatVec(b, x, y)
}

func silu(x float32) float32 {
	return x / (1 + float32(math.Exp(float64(-x))))
}

func softplus(x float64) float64 {
	if x > 20 {
		return x
	}
	return math.Log1p(math.Exp(x))
}

func l2normInPlace(x []float32, eps float64) {
	var s float64
	for _, v := range x {
		s += float64(v) * float64(v)
	}
	inv := float32(1 / math.Sqrt(s+eps))
	for i := range x {
		x[i] *= inv
	}
}

func rmsNormGated(x, z, gamma []float32, eps float32) {
	var mean float32
	for _, v := range x {
		mean += v * v
	}
	mean /= float32(len(x))
	inv := 1 / float32(math.Sqrt(float64(mean+eps)))
	for i := range x {
		g := float32(1)
		if i < len(gamma) {
			g = gamma[i]
		}
		x[i] = x[i] * inv * g * silu(z[i])
	}
}
