package flux2

import "math"

// RotaryEmb is real-valued ND RoPE freqs (cos, sin) each [seq * ropeDim].
// ropeDim = sum(axes_dims_rope) = attention_head_dim (128 for Klein).
type RotaryEmb struct {
	Cos, Sin []float32
	Seq      int
	Dim      int // head dim
}

// Get1DRotaryPosEmbed ports diffusers get_1d_rotary_pos_embed with use_real +
// repeat_interleave_real (Flux-style).
// pos length = seq; returns cos,sin each [seq * dim].
func Get1DRotaryPosEmbed(dim int, pos []float64, theta float64) (cos, sin []float32) {
	if dim%2 != 0 {
		panic("Get1DRotaryPosEmbed: dim must be even")
	}
	seq := len(pos)
	half := dim / 2
	freqs := make([]float64, half)
	for i := 0; i < half; i++ {
		freqs[i] = 1.0 / math.Pow(theta, float64(i)/float64(dim))
	}
	// outer(pos, freqs) → [seq, half], then cos/sin with repeat_interleave(2)
	cos = make([]float32, seq*dim)
	sin = make([]float32, seq*dim)
	for s := 0; s < seq; s++ {
		for i := 0; i < half; i++ {
			angle := pos[s] * freqs[i]
			c := float32(math.Cos(angle))
			sn := float32(math.Sin(angle))
			// repeat_interleave: [c0,c0,c1,c1,...]
			cos[s*dim+2*i] = c
			cos[s*dim+2*i+1] = c
			sin[s*dim+2*i] = sn
			sin[s*dim+2*i+1] = sn
		}
	}
	return cos, sin
}

// PosEmbedND computes Flux2 ND RoPE from ids [seq * nAxes] row-major.
// axesDims length must match nAxes; sum(axesDims) == headDim.
func PosEmbedND(ids []float32, seq int, axesDims []int, theta float64) RotaryEmb {
	nAxes := len(axesDims)
	headDim := 0
	for _, d := range axesDims {
		headDim += d
	}
	cosOut := make([]float32, seq*headDim)
	sinOut := make([]float32, seq*headDim)
	col := 0
	for a := 0; a < nAxes; a++ {
		pos := make([]float64, seq)
		for s := 0; s < seq; s++ {
			pos[s] = float64(ids[s*nAxes+a])
		}
		c, sn := Get1DRotaryPosEmbed(axesDims[a], pos, theta)
		for s := 0; s < seq; s++ {
			copy(cosOut[s*headDim+col:s*headDim+col+axesDims[a]], c[s*axesDims[a]:(s+1)*axesDims[a]])
			copy(sinOut[s*headDim+col:s*headDim+col+axesDims[a]], sn[s*axesDims[a]:(s+1)*axesDims[a]])
		}
		col += axesDims[a]
	}
	return RotaryEmb{Cos: cosOut, Sin: sinOut, Seq: seq, Dim: headDim}
}

// ConcatRotary concatenates text then image RoPE along the sequence axis.
func ConcatRotary(txt, img RotaryEmb) RotaryEmb {
	if txt.Dim != img.Dim {
		panic("ConcatRotary: dim mismatch")
	}
	seq := txt.Seq + img.Seq
	dim := txt.Dim
	cos := make([]float32, seq*dim)
	sin := make([]float32, seq*dim)
	copy(cos, txt.Cos)
	copy(sin, txt.Sin)
	copy(cos[txt.Seq*dim:], img.Cos)
	copy(sin[txt.Seq*dim:], img.Sin)
	return RotaryEmb{Cos: cos, Sin: sin, Seq: seq, Dim: dim}
}

// ApplyRotaryEmb applies Flux-style real RoPE to x laid out [seq, heads, headDim]
// (flattened). Matches apply_rotary_emb(..., use_real_unbind_dim=-1, sequence_dim=1).
func ApplyRotaryEmb(x []float32, rope RotaryEmb, seq, heads, headDim int) {
	if rope.Seq < seq || rope.Dim != headDim {
		panic("ApplyRotaryEmb: rope shape mismatch")
	}
	for s := 0; s < seq; s++ {
		cos := rope.Cos[s*headDim : (s+1)*headDim]
		sin := rope.Sin[s*headDim : (s+1)*headDim]
		for h := 0; h < heads; h++ {
			base := (s*heads+h)*headDim
			// rotate pairs: x_rotated = [-x1, x0, -x3, x2, ...]
			for d := 0; d+1 < headDim; d += 2 {
				x0 := x[base+d]
				x1 := x[base+d+1]
				// out = x * cos + rotated * sin
				x[base+d] = x0*cos[d] + (-x1)*sin[d]
				x[base+d+1] = x1*cos[d+1] + x0*sin[d+1]
			}
		}
	}
}
