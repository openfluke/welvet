package mosstts

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/simd"
	"github.com/openfluke/welvet/webgpu"
)

type patchedPretransform struct {
	patch      int
	downsample bool
}

func (p *patchedPretransform) Forward(x []float32, channels, length int) ([]float32, int, int) {
	h := p.patch
	if p.downsample {
		// encode: (B,d,L) → (B,d*h,L/h)
		outLen := length / h
		outCh := channels * h
		out := make([]float32, outCh*outLen)
		for c := 0; c < channels; c++ {
			for t := 0; t < outLen; t++ {
				for kk := 0; kk < h; kk++ {
					// reshape d,L → d,L/h,h → permute → d,h,L/h → d*h,L/h
					src := x[c*length+(t*h+kk)]
					out[(c*h+kk)*outLen+t] = src
				}
			}
		}
		return out, outCh, outLen
	}
	// decode upsample: (B, d*h, L) → (B, d, L*h)
	if channels%h != 0 {
		// best-effort
	}
	d := channels / h
	outLen := length * h
	out := make([]float32, d*outLen)
	for c := 0; c < d; c++ {
		for t := 0; t < length; t++ {
			for kk := 0; kk < h; kk++ {
				src := x[(c*h+kk)*length+t]
				out[c*outLen+(t*h+kk)] = src
			}
		}
	}
	return out, d, outLen
}

type projectedTransformer struct {
	InProjW, OutProjW []float32
	InDim, OutDim     int
	DModel            int
	Layers            []codecTFLayer
	NumHeads          int
	Context           int
	Causal            bool
	MaxPeriod         float64
	UseSIMD, UseGPU   bool
}

type codecTFLayer struct {
	Norm1W, Norm1B, Norm2W, Norm2B []float32
	InProjW, OutProjW             []float32 // attn
	FFN0W, FFN2W                  []float32
	Scale1, Scale2                []float32
	DModel, FF                    int
	NumHeads                      int
	UseSIMD, UseGPU               bool
}

func loadProjectedTransformer(tensors map[string][]float32, idx int, kw map[string]any, context int) (*projectedTransformer, error) {
	prefix := fmt.Sprintf("decoder.%d", idx)
	inDim := intFromAny(kw["input_dimension"], 0)
	outDim := intFromAny(kw["output_dimension"], 0)
	dModel := intFromAny(kw["d_model"], 256)
	nHeads := intFromAny(kw["num_heads"], 4)
	nLayers := intFromAny(kw["num_layers"], 4)
	ff := intFromAny(kw["dim_feedforward"], 1024)
	maxPeriod := floatFromAny(kw["max_period"], 10000)
	causal := boolFromAny(kw["causal"], true)

	inW, ok := tensors[prefix+".input_proj.weight"]
	if !ok {
		return nil, fmt.Errorf("missing %s.input_proj.weight", prefix)
	}
	outW, ok := tensors[prefix+".output_proj.weight"]
	if !ok {
		return nil, fmt.Errorf("missing %s.output_proj.weight", prefix)
	}

	layers := make([]codecTFLayer, nLayers)
	for i := 0; i < nLayers; i++ {
		lp := fmt.Sprintf("%s.transformer.layers.%d", prefix, i)
		get := func(suf string) ([]float32, error) {
			v, ok := tensors[lp+"."+suf]
			if !ok {
				return nil, fmt.Errorf("missing %s.%s", lp, suf)
			}
			return v, nil
		}
		var err error
		layers[i].Norm1W, err = get("norm1.weight")
		if err != nil {
			return nil, err
		}
		layers[i].Norm1B, err = get("norm1.bias")
		if err != nil {
			return nil, err
		}
		layers[i].Norm2W, err = get("norm2.weight")
		if err != nil {
			return nil, err
		}
		layers[i].Norm2B, err = get("norm2.bias")
		if err != nil {
			return nil, err
		}
		layers[i].InProjW, err = get("self_attn.in_proj.weight")
		if err != nil {
			return nil, err
		}
		layers[i].OutProjW, err = get("self_attn.out_proj.weight")
		if err != nil {
			return nil, err
		}
		layers[i].FFN0W, err = get("ffn.0.weight")
		if err != nil {
			return nil, err
		}
		layers[i].FFN2W, err = get("ffn.2.weight")
		if err != nil {
			return nil, err
		}
		layers[i].Scale1 = tensors[lp+".layer_scale_1.scale"]
		layers[i].Scale2 = tensors[lp+".layer_scale_2.scale"]
		layers[i].DModel = dModel
		layers[i].FF = ff
		layers[i].NumHeads = nHeads
	}

	return &projectedTransformer{
		InProjW: inW, OutProjW: outW,
		InDim: inDim, OutDim: outDim, DModel: dModel,
		Layers: layers, NumHeads: nHeads, Context: context,
		Causal: causal, MaxPeriod: maxPeriod,
	}, nil
}

func (m *projectedTransformer) Forward(x []float32, channels, length int) ([]float32, int, int) {
	// x: [channels][length] → treat as (B=1,C,T); project to (T, d_model)
	h := make([]float32, length*m.DModel)
	for t := 0; t < length; t++ {
		tok := make([]float32, channels)
		for c := 0; c < channels; c++ {
			tok[c] = x[c*length+t]
		}
		// Linear in_proj: W [d_model, in_dim]
		dst := h[t*m.DModel : (t+1)*m.DModel]
		fusedLinear(m.InProjW, m.DModel, channels, tok, dst, m.UseSIMD, m.UseGPU)
	}
	for i := range m.Layers {
		m.Layers[i].forward(h, length, m.Causal, m.Context, m.MaxPeriod)
	}
	out := make([]float32, m.OutDim*length)
	for t := 0; t < length; t++ {
		src := h[t*m.DModel : (t+1)*m.DModel]
		dst := make([]float32, m.OutDim)
		fusedLinear(m.OutProjW, m.OutDim, m.DModel, src, dst, m.UseSIMD, m.UseGPU)
		for o := 0; o < m.OutDim; o++ {
			out[o*length+t] = dst[o]
		}
	}
	return out, m.OutDim, length
}

func fusedLinear(w []float32, out, in int, x, y []float32, useSIMD, useGPU bool) {
	const gpuMinElems = 512 * 1024
	if useGPU && webgpu.Available() && out*in >= gpuMinElems && len(w) >= out*in {
		key := webgpu.BlobKey(unsafe.Pointer(&w[0]))
		if err := webgpu.DenseGEMVF32Resident(key, w, x, y, 1, in, out); err == nil {
			return
		}
	}
	if (useSIMD || useGPU) && simd.Enabled() {
		dense.GemvF32SIMD(w, x, y, out, in)
		return
	}
	linearNoBias(w, out, in, x, y)
}

func linearNoBias(w []float32, out, in int, x, y []float32) {
	for o := 0; o < out; o++ {
		var acc float32
		row := w[o*in : (o+1)*in]
		for i := 0; i < in; i++ {
			acc += row[i] * x[i]
		}
		y[o] = acc
	}
}

func (m *projectedTransformer) setFuse(simdOn, gpuOn bool) {
	m.UseSIMD, m.UseGPU = simdOn, gpuOn
	for i := range m.Layers {
		m.Layers[i].UseSIMD, m.Layers[i].UseGPU = simdOn, gpuOn
	}
}

func (m *projectedTransformer) warmGPU() (n int, err error) {
	warm := func(w []float32, rows, cols int) error {
		if len(w) < rows*cols {
			return nil
		}
		const gpuMinElems = 512 * 1024
		if rows*cols < gpuMinElems {
			return nil
		}
		return webgpu.WarmF32Weight(webgpu.BlobKey(unsafe.Pointer(&w[0])), w, rows, cols)
	}
	if e := warm(m.InProjW, m.DModel, m.InDim); e != nil {
		if webgpu.IsF32VRAMFull(e) {
			return n, nil
		}
		return n, e
	}
	n++
	if e := warm(m.OutProjW, m.OutDim, m.DModel); e != nil {
		if webgpu.IsF32VRAMFull(e) {
			return n, nil
		}
		return n, e
	}
	n++
	for i := range m.Layers {
		l := &m.Layers[i]
		d, ff := l.DModel, l.FF
		for _, it := range []struct {
			w          []float32
			rows, cols int
		}{
			{l.InProjW, 3 * d, d},
			{l.OutProjW, d, d},
			{l.FFN0W, ff, d},
			{l.FFN2W, d, ff},
		} {
			if e := warm(it.w, it.rows, it.cols); e != nil {
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

func (l *codecTFLayer) forward(h []float32, length int, causal bool, context int, maxPeriod float64) {
	d := l.DModel
	heads := l.NumHeads
	headDim := d / heads

	normed := make([]float32, length*d)
	copy(normed, h)
	for t := 0; t < length; t++ {
		layerNorm(normed[t*d:(t+1)*d], l.Norm1W, l.Norm1B, d, 1e-5)
	}

	// QKV: in_proj [3d, d]
	qkv := make([]float32, length*3*d)
	for t := 0; t < length; t++ {
		fusedLinear(l.InProjW, 3*d, d, normed[t*d:(t+1)*d], qkv[t*3*d:(t+1)*3*d], l.UseSIMD, l.UseGPU)
	}
	// reshape to heads: for each t, split q,k,v
	q := make([]float32, length*d)
	k := make([]float32, length*d)
	v := make([]float32, length*d)
	for t := 0; t < length; t++ {
		copy(q[t*d:(t+1)*d], qkv[t*3*d:t*3*d+d])
		copy(k[t*d:(t+1)*d], qkv[t*3*d+d:t*3*d+2*d])
		copy(v[t*d:(t+1)*d], qkv[t*3*d+2*d:t*3*d+3*d])
	}
	applyCodecRoPE(q, k, length, heads, headDim, maxPeriod)

	attnOut := make([]float32, length*d)
	scale := float32(1 / math.Sqrt(float64(headDim)))
	for head := 0; head < heads; head++ {
		for t := 0; t < length; t++ {
			scores := make([]float32, length)
			rowMax := float32(-math.MaxFloat32)
			qOff := t*d + head*headDim
			qi := q[qOff : qOff+headDim]
			for s := 0; s < length; s++ {
				if causal && s > t {
					scores[s] = float32(-1e30)
					continue
				}
				if context > 0 && t-s >= context {
					scores[s] = float32(-1e30)
					continue
				}
				kOff := s*d + head*headDim
				kj := k[kOff : kOff+headDim]
				var dot float32
				if simd.Enabled() {
					dot = float32(simd.DotTile(qi, kj, 0, headDim, 0))
				} else {
					for i := 0; i < headDim; i++ {
						dot += qi[i] * kj[i]
					}
				}
				scores[s] = dot * scale
				if scores[s] > rowMax {
					rowMax = scores[s]
				}
			}
			var sum float32
			for s := 0; s < length; s++ {
				e := float32(math.Exp(float64(scores[s] - rowMax)))
				scores[s] = e
				sum += e
			}
			inv := float32(1)
			if sum > 0 {
				inv = 1 / sum
			}
			acc := make([]float32, headDim)
			for s := 0; s < length; s++ {
				w := scores[s] * inv
				vOff := s*d + head*headDim
				vj := v[vOff : vOff+headDim]
				for i := 0; i < headDim; i++ {
					acc[i] += w * vj[i]
				}
			}
			copy(attnOut[qOff:qOff+headDim], acc)
		}
	}
	proj := make([]float32, length*d)
	for t := 0; t < length; t++ {
		fusedLinear(l.OutProjW, d, d, attnOut[t*d:(t+1)*d], proj[t*d:(t+1)*d], l.UseSIMD, l.UseGPU)
	}
	for t := 0; t < length; t++ {
		for i := 0; i < d; i++ {
			val := proj[t*d+i]
			if l.Scale1 != nil {
				val *= l.Scale1[i]
			}
			h[t*d+i] += val
		}
	}

	copy(normed, h)
	for t := 0; t < length; t++ {
		layerNorm(normed[t*d:(t+1)*d], l.Norm2W, l.Norm2B, d, 1e-5)
	}
	mid := make([]float32, l.FF)
	out := make([]float32, d)
	for t := 0; t < length; t++ {
		fusedLinear(l.FFN0W, l.FF, d, normed[t*d:(t+1)*d], mid, l.UseSIMD, l.UseGPU)
		for i := range mid {
			mid[i] = gelu(mid[i])
		}
		fusedLinear(l.FFN2W, d, l.FF, mid, out, l.UseSIMD, l.UseGPU)
		for i := 0; i < d; i++ {
			val := out[i]
			if l.Scale2 != nil {
				val *= l.Scale2[i]
			}
			h[t*d+i] += val
		}
	}
}

func gelu(x float32) float32 {
	// standard GELU approx
	xf := float64(x)
	return float32(0.5 * xf * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(xf+0.044715*xf*xf*xf))))
}

// applyCodecRoPE matches MossAudioTokenizer apply_rope (time_before_heads=False): q,k as [T, H*D] interleaved heads.
func applyCodecRoPE(q, k []float32, length, heads, headDim int, maxPeriod float64) {
	half := headDim / 2
	freqs := make([]float64, half)
	for i := 0; i < half; i++ {
		freqs[i] = math.Exp(float64(i) * (-math.Log(maxPeriod) * 2 / float64(headDim)))
	}
	for t := 0; t < length; t++ {
		for head := 0; head < heads; head++ {
			off := t*(heads*headDim) + head*headDim
			rotateCodec(q[off:off+headDim], float64(t), freqs)
			rotateCodec(k[off:off+headDim], float64(t), freqs)
		}
	}
}

func rotateCodec(x []float32, pos float64, freqs []float64) {
	half := len(x) / 2
	// view as pairs (re, im) along last dim
	for i := 0; i < half; i++ {
		re := float64(x[2*i])
		im := float64(x[2*i+1])
		angle := freqs[i] * pos
		c, s := math.Cos(angle), math.Sin(angle)
		x[2*i] = float32(re*c - im*s)
		x[2*i+1] = float32(re*s + im*c)
	}
}
