package flux2

import "math"

// RMSNorm applies root-mean-square normalization over the last dim elements.
// If weight is nil, this is a pure RMS (no affine). weight length must be dim when set.
func RMSNorm(x []float32, weight []float32, dim int, eps float64) {
	if dim <= 0 || len(x) < dim {
		return
	}
	var sumSq float64
	for i := 0; i < dim; i++ {
		v := float64(x[i])
		sumSq += v * v
	}
	inv := 1.0 / math.Sqrt(sumSq/float64(dim)+eps)
	if weight == nil {
		for i := 0; i < dim; i++ {
			x[i] = float32(float64(x[i]) * inv)
		}
		return
	}
	for i := 0; i < dim; i++ {
		x[i] = float32(float64(x[i]) * inv * float64(weight[i]))
	}
}

// RMSNormSeq applies RMSNorm to each of seq tokens in x [seq*dim].
func RMSNormSeq(x, weight []float32, seq, dim int, eps float64) {
	for s := 0; s < seq; s++ {
		RMSNorm(x[s*dim:(s+1)*dim], weight, dim, eps)
	}
}

// LayerNormNoAffine is LayerNorm without learnable affine (Flux2 double/single block norms).
func LayerNormNoAffine(x []float32, dim int, eps float64) {
	if dim <= 0 || len(x) < dim {
		return
	}
	var mean float64
	for i := 0; i < dim; i++ {
		mean += float64(x[i])
	}
	mean /= float64(dim)
	var varSum float64
	for i := 0; i < dim; i++ {
		d := float64(x[i]) - mean
		varSum += d * d
	}
	inv := 1.0 / math.Sqrt(varSum/float64(dim)+eps)
	for i := 0; i < dim; i++ {
		x[i] = float32((float64(x[i]) - mean) * inv)
	}
}

// LayerNormNoAffineSeq applies LayerNormNoAffine over seq tokens.
func LayerNormNoAffineSeq(x []float32, seq, dim int, eps float64) {
	for s := 0; s < seq; s++ {
		LayerNormNoAffine(x[s*dim:(s+1)*dim], dim, eps)
	}
}

// ModTriple is (shift, scale, gate) for AdaLN-style modulation.
type ModTriple struct {
	Shift, Scale, Gate []float32 // each length dim (broadcast over seq)
}

// SplitModulation splits a flat mod vector of length dim*3*sets into sets of ModTriple.
// Matches Flux2Modulation.split: chunks of (shift, scale, gate) along the last dim.
func SplitModulation(mod []float32, dim, sets int) []ModTriple {
	out := make([]ModTriple, sets)
	for i := 0; i < sets; i++ {
		base := i * 3 * dim
		out[i] = ModTriple{
			Shift: mod[base : base+dim],
			Scale: mod[base+dim : base+2*dim],
			Gate:  mod[base+2*dim : base+3*dim],
		}
	}
	return out
}

// ApplyModulate: y = (1+scale)*x + shift  (in-place on x for each token).
func ApplyModulate(x, shift, scale []float32, seq, dim int) {
	for s := 0; s < seq; s++ {
		off := s * dim
		for d := 0; d < dim; d++ {
			x[off+d] = (1+scale[d])*x[off+d] + shift[d]
		}
	}
}

// ApplyGateResidual: dst = dst + gate * src  (per token).
func ApplyGateResidual(dst, src, gate []float32, seq, dim int) {
	for s := 0; s < seq; s++ {
		off := s * dim
		for d := 0; d < dim; d++ {
			dst[off+d] += gate[d] * src[off+d]
		}
	}
}

// AdaLayerNormContinuous: LayerNorm(x) * (1+scale) + shift from SiLU(temb)→Linear→chunk.
func AdaLayerNormContinuous(x, scale, shift []float32, seq, dim int, eps float64) {
	for s := 0; s < seq; s++ {
		off := s * dim
		LayerNormNoAffine(x[off:off+dim], dim, eps)
		for d := 0; d < dim; d++ {
			x[off+d] = x[off+d]*(1+scale[d]) + shift[d]
		}
	}
}

// SiLU applies x * sigmoid(x) in-place.
func SiLU(x []float32) {
	for i, v := range x {
		x[i] = v / float32(1+math.Exp(float64(-v)))
	}
}

// SiLUCopy returns silu(x) without mutating x.
func SiLUCopy(x []float32) []float32 {
	out := make([]float32, len(x))
	copy(out, x)
	SiLU(out)
	return out
}
