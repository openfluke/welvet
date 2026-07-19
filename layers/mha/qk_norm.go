package mha

import "math"

// applyQKNormInPlace does per-head RMSNorm on a packed [numHeads*headDim] vector.
func applyQKNormInPlace(vec, gamma []float64, numHeads, headDim int, eps float64) {
	if numHeads <= 0 || headDim <= 0 || len(vec) < numHeads*headDim {
		return
	}
	if eps <= 0 {
		eps = 1e-6
	}
	for h := 0; h < numHeads; h++ {
		start := h * headDim
		var sumSq float64
		for d := 0; d < headDim; d++ {
			v := vec[start+d]
			sumSq += v * v
		}
		inv := 1.0 / math.Sqrt(sumSq/float64(headDim)+eps)
		for d := 0; d < headDim; d++ {
			scale := 1.0
			if len(gamma) == headDim {
				scale = gamma[d]
			} else if len(gamma) == numHeads*headDim {
				scale = gamma[h*headDim+d]
			}
			vec[start+d] *= inv * scale
		}
	}
}
