package mosstts

import (
	"math"
	"unsafe"

	"github.com/openfluke/welvet/dense"
	"github.com/openfluke/welvet/simd"
	"github.com/openfluke/welvet/webgpu"
)

func geluNew(x float32) float32 {
	// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
	const k = 0.7978845608028654 // sqrt(2/pi)
	xf := float64(x)
	inner := k * (xf + 0.044715*xf*xf*xf)
	return float32(0.5 * xf * (1 + math.Tanh(inner)))
}

func layerNorm(x, weight, bias []float32, dim int, eps float64) {
	var sum, sumSq float64
	for i := 0; i < dim; i++ {
		v := float64(x[i])
		sum += v
		sumSq += v * v
	}
	mean := sum / float64(dim)
	var_ := sumSq/float64(dim) - mean*mean
	inv := 1 / math.Sqrt(var_+eps)
	for i := 0; i < dim; i++ {
		v := (float64(x[i]) - mean) * inv
		x[i] = float32(v)*weight[i] + bias[i]
	}
}

func rotateHalf(x []float32) {
	// interleaved: [-x1, x0, -x3, x2, ...]
	tmp := make([]float32, len(x))
	copy(tmp, x)
	for i := 0; i+1 < len(x); i += 2 {
		x[i] = -tmp[i+1]
		x[i+1] = tmp[i]
	}
}

func applyRoPE(qOrK []float32, cos, sin []float32) {
	// qOrK, cos, sin length = headDim
	rotated := make([]float32, len(qOrK))
	copy(rotated, qOrK)
	rotateHalf(rotated)
	for i := range qOrK {
		qOrK[i] = qOrK[i]*cos[i] + rotated[i]*sin[i]
	}
}

func ropeCosSin(pos int, headDim int, base float64, outCos, outSin []float32) {
	// match MossTTSNanoGPT2RotaryEmbedding: inv_freq on even dims, repeat_interleave 2
	half := headDim / 2
	for i := 0; i < half; i++ {
		freq := 1.0 / math.Pow(base, float64(2*i)/float64(headDim))
		angle := float64(pos) * freq
		c, s := float32(math.Cos(angle)), float32(math.Sin(angle))
		outCos[2*i] = c
		outCos[2*i+1] = c
		outSin[2*i] = s
		outSin[2*i+1] = s
	}
}

// Linear dense [out,in] row-major with optional SIMD / sticky GPU fuse.
type Linear struct {
	Out, In int
	W       []float32
	B       []float32 // optional
	UseSIMD bool
	UseGPU  bool
}

func (l *Linear) gpuKey() uintptr {
	if l == nil || len(l.W) == 0 {
		return 0
	}
	return webgpu.BlobKey(unsafe.Pointer(&l.W[0]))
}

func (l *Linear) Forward(x, y []float32) {
	// Per-GEMV WebGPU is not fuse — AR GPU path uses gpt2Fuse (one submit/token).
	// Linear stays on SIMD/host; UseGPU here only selects SIMD when available.
	if (l.UseSIMD || l.UseGPU) && simd.Enabled() {
		dense.GemvF32SIMD(l.W, x, y, l.Out, l.In)
		if l.B != nil {
			for o := 0; o < l.Out; o++ {
				y[o] += l.B[o]
			}
		}
		return
	}
	for o := 0; o < l.Out; o++ {
		var acc float32
		row := l.W[o*l.In : (o+1)*l.In]
		for i := 0; i < l.In; i++ {
			acc += row[i] * x[i]
		}
		if l.B != nil {
			acc += l.B[o]
		}
		y[o] = acc
	}
}

// WarmGPU uploads this linear's weights to sticky VRAM when large enough for GPU fuse.
func (l *Linear) WarmGPU() error {
	if l == nil || len(l.W) < l.Out*l.In {
		return nil
	}
	const gpuMinElems = 512 * 1024
	if l.Out*l.In < gpuMinElems {
		return nil
	}
	return webgpu.WarmF32Weight(l.gpuKey(), l.W, l.Out, l.In)
}

type gpt2Attn struct {
	NumHeads, HeadDim, Hidden int
	CAttn, CProj              Linear
	Scale                     bool
	RopeBase                  float64
}

type kvCache struct {
	K, V []float32 // [seq, heads*headDim] growing
	Seq  int
}

func (a *gpt2Attn) forward(x []float32, mask []bool, posOffset int, cache *kvCache) []float32 {
	h := a.Hidden
	qkv := make([]float32, 3*h)
	a.CAttn.Forward(x, qkv)
	q := qkv[0:h]
	k := qkv[h : 2*h]
	v := qkv[2*h : 3*h]

	// RoPE per head
	cos := make([]float32, a.HeadDim)
	sin := make([]float32, a.HeadDim)
	pos := posOffset
	if cache != nil {
		pos = cache.Seq + posOffset
	}
	ropeCosSin(pos, a.HeadDim, a.RopeBase, cos, sin)
	for head := 0; head < a.NumHeads; head++ {
		off := head * a.HeadDim
		applyRoPE(q[off:off+a.HeadDim], cos, sin)
		applyRoPE(k[off:off+a.HeadDim], cos, sin)
	}

	// append to cache
	var keys, vals []float32
	if cache != nil {
		cache.K = append(cache.K, k...)
		cache.V = append(cache.V, v...)
		cache.Seq++
		keys, vals = cache.K, cache.V
	} else {
		keys, vals = k, v
	}
	keySeq := len(keys) / h

	out := make([]float32, h)
	scale := float32(1)
	if a.Scale {
		scale = 1 / float32(math.Sqrt(float64(a.HeadDim)))
	}
	for head := 0; head < a.NumHeads; head++ {
		qOff := head * a.HeadDim
		qi := q[qOff : qOff+a.HeadDim]
		scores := make([]float32, keySeq)
		rowMax := float32(-math.MaxFloat32)
		for t := 0; t < keySeq; t++ {
			if mask != nil && t < len(mask) && !mask[t] {
				scores[t] = float32(-1e30)
				continue
			}
			kj := keys[t*h+qOff : t*h+qOff+a.HeadDim]
			var dot float32
			if simd.Enabled() {
				dot = float32(simd.DotTile(qi, kj, 0, a.HeadDim, 0))
			} else {
				for d := 0; d < a.HeadDim; d++ {
					dot += qi[d] * kj[d]
				}
			}
			s := dot * scale
			scores[t] = s
			if s > rowMax {
				rowMax = s
			}
		}
		var sum float32
		for t := 0; t < keySeq; t++ {
			e := float32(math.Exp(float64(scores[t] - rowMax)))
			scores[t] = e
			sum += e
		}
		inv := 1 / sum
		acc := make([]float32, a.HeadDim)
		for t := 0; t < keySeq; t++ {
			w := scores[t] * inv
			vj := vals[t*h+qOff : t*h+qOff+a.HeadDim]
			for d := 0; d < a.HeadDim; d++ {
				acc[d] += w * vj[d]
			}
		}
		copy(out[qOff:qOff+a.HeadDim], acc)
	}
	proj := make([]float32, h)
	a.CProj.Forward(out, proj)
	return proj
}

type gpt2Block struct {
	LN1W, LN1B, LN2W, LN2B []float32
	Attn                   gpt2Attn
	FcIn, FcOut            Linear
	Eps                    float64
	Hidden                 int
}

func (b *gpt2Block) forward(x []float32, mask []bool, pos int, cache *kvCache) {
	h := b.Hidden
	normed := make([]float32, h)
	copy(normed, x)
	layerNorm(normed, b.LN1W, b.LN1B, h, b.Eps)
	attnOut := b.Attn.forward(normed, mask, pos, cache)
	for i := 0; i < h; i++ {
		x[i] += attnOut[i]
	}
	copy(normed, x)
	layerNorm(normed, b.LN2W, b.LN2B, h, b.Eps)
	mid := make([]float32, b.FcIn.Out)
	b.FcIn.Forward(normed, mid)
	for i := range mid {
		mid[i] = geluNew(mid[i])
	}
	out := make([]float32, h)
	b.FcOut.Forward(mid, out)
	for i := 0; i < h; i++ {
		x[i] += out[i]
	}
}

// GPT2Model is MossTTSNanoGPT2Model (wte optional Identity for local).
type GPT2Model struct {
	WTE     []float32 // [vocab*hidden] or nil if Identity
	Vocab   int
	Hidden  int
	Blocks  []gpt2Block
	LNFW    []float32
	LNFB    []float32
	Eps     float64
	HasWTE  bool
}

func (m *GPT2Model) embedToken(id int, dst []float32) {
	if !m.HasWTE {
		return
	}
	copy(dst, m.WTE[id*m.Hidden:(id+1)*m.Hidden])
}

// ForwardSeq runs full sequence (no KV cache). x is [seq*hidden] in/out.
func (m *GPT2Model) ForwardSeq(x []float32, seq int, attnMask []bool) {
	h := m.Hidden
	caches := make([]kvCache, len(m.Blocks))
	for pos := 0; pos < seq; pos++ {
		tok := x[pos*h : (pos+1)*h]
		var mask []bool
		if attnMask != nil {
			mask = attnMask[:pos+1]
		}
		for li := range m.Blocks {
			m.Blocks[li].forward(tok, mask, 0, &caches[li])
		}
	}
	for pos := 0; pos < seq; pos++ {
		tok := x[pos*h : (pos+1)*h]
		layerNorm(tok, m.LNFW, m.LNFB, h, m.Eps)
	}
}

// ForwardLast runs one new token with KV caches (len = n_layer). Returns updated hidden for that token.
func (m *GPT2Model) ForwardLast(x []float32, caches []kvCache, attnMask []bool) {
	for li := range m.Blocks {
		m.Blocks[li].forward(x, attnMask, 0, &caches[li])
	}
	layerNorm(x, m.LNFW, m.LNFB, m.Hidden, m.Eps)
}

// ForwardLocal runs local transformer over growing embeds [seq*hidden] without KV (small seq).
func (m *GPT2Model) ForwardLocal(embeds []float32, seq int) []float32 {
	h := m.Hidden
	x := make([]float32, len(embeds))
	copy(x, embeds)
	caches := make([]kvCache, len(m.Blocks))
	for pos := 0; pos < seq; pos++ {
		tok := x[pos*h : (pos+1)*h]
		for li := range m.Blocks {
			m.Blocks[li].forward(tok, nil, 0, &caches[li])
		}
	}
	last := make([]float32, h)
	copy(last, x[(seq-1)*h:seq*h])
	layerNorm(last, m.LNFW, m.LNFB, h, m.Eps)
	return last
}

// DecodeStep runs one new token through blocks with KV caches (no final LN).
func (m *GPT2Model) DecodeStep(x []float32, caches []kvCache) {
	for li := range m.Blocks {
		m.Blocks[li].forward(x, nil, 0, &caches[li])
	}
}

// FinalNorm applies ln_f in-place.
func (m *GPT2Model) FinalNorm(x []float32) {
	layerNorm(x, m.LNFW, m.LNFB, m.Hidden, m.Eps)
}

func (m *GPT2Model) setFuse(simdOn, gpuOn bool) {
	if m == nil {
		return
	}
	for i := range m.Blocks {
		b := &m.Blocks[i]
		b.Attn.CAttn.UseSIMD, b.Attn.CAttn.UseGPU = simdOn, gpuOn
		b.Attn.CProj.UseSIMD, b.Attn.CProj.UseGPU = simdOn, gpuOn
		b.FcIn.UseSIMD, b.FcIn.UseGPU = simdOn, gpuOn
		b.FcOut.UseSIMD, b.FcOut.UseGPU = simdOn, gpuOn
	}
}

func (m *GPT2Model) warmGPU() (n int, err error) {
	if m == nil {
		return 0, nil
	}
	for i := range m.Blocks {
		b := &m.Blocks[i]
		for _, l := range []*Linear{&b.Attn.CAttn, &b.Attn.CProj, &b.FcIn, &b.FcOut} {
			if e := l.WarmGPU(); e != nil {
				if webgpu.IsF32VRAMFull(e) {
					return n, nil
				}
				return n, e
			}
			n++
		}
	}
	return n, nil
}
