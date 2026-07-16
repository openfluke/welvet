package mha

import "math"

// applyRoPE rotates pairs in-place for numHeads heads packed in vec (len >= numHeads*headDim).
func applyRoPE(vec []float64, pos, numHeads, headDim int, theta float64) {
	half := headDim / 2
	if half <= 0 || theta <= 0 {
		return
	}
	for h := 0; h < numHeads; h++ {
		base := h * headDim
		for d := 0; d < half; d++ {
			angle := float64(pos) / math.Pow(theta, float64(2*d)/float64(headDim))
			c, s := math.Cos(angle), math.Sin(angle)
			v0, v1 := vec[base+d], vec[base+d+half]
			vec[base+d] = v0*c - v1*s
			vec[base+d+half] = v0*s + v1*c
		}
	}
}

// applyRoPEBackward applies the transpose of the RoPE rotation to grads (in-place).
func applyRoPEBackward(grad []float64, pos, numHeads, headDim int, theta float64) {
	half := headDim / 2
	if half <= 0 || theta <= 0 {
		return
	}
	for h := 0; h < numHeads; h++ {
		base := h * headDim
		for d := 0; d < half; d++ {
			angle := float64(pos) / math.Pow(theta, float64(2*d)/float64(headDim))
			c, s := math.Cos(angle), math.Sin(angle)
			v0, v1 := grad[base+d], grad[base+d+half]
			grad[base+d] = v0*c + v1*s
			grad[base+d+half] = -v0*s + v1*c
		}
	}
}
