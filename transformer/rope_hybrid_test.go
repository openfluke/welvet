package transformer

import (
	"math"
	"testing"
)

// Qwen3 / Lucy use rotate_half (pair d with d+half), not adjacent NeoX pairs.
func TestApplyPartialRoPERotateHalf(t *testing.T) {
	headDim := 8
	x := make([]float32, headDim)
	for i := range x {
		x[i] = float32(i + 1)
	}
	applyPartialRoPE(x, 1, headDim, 1.0, 10000, 1)

	// Manual rotate_half reference at pos=1.
	ref := make([]float32, headDim)
	for i := range ref {
		ref[i] = float32(i + 1)
	}
	half := headDim / 2
	for d := 0; d < half; d++ {
		freq := 1 / math.Pow(10000, float64(2*d)/float64(headDim))
		ang := 1.0 * freq
		c, s := float32(math.Cos(ang)), float32(math.Sin(ang))
		v0, v1 := ref[d], ref[d+half]
		ref[d] = v0*c - v1*s
		ref[d+half] = v0*s + v1*c
	}
	for i := range x {
		if math.Abs(float64(x[i]-ref[i])) > 1e-5 {
			t.Fatalf("dim %d: got %g want %g (rotate_half)", i, x[i], ref[i])
		}
	}
	// Adjacent pairing would move x[0] with x[1] — must differ at pos>0.
	adj := make([]float32, headDim)
	for i := range adj {
		adj[i] = float32(i + 1)
	}
	for i := 0; i < headDim; i += 2 {
		freq := 1 / math.Pow(10000, float64(i)/float64(headDim))
		ang := 1.0 * freq
		c, s := float32(math.Cos(ang)), float32(math.Sin(ang))
		u, v := adj[i], adj[i+1]
		adj[i] = u*c - v*s
		adj[i+1] = u*s + v*c
	}
	differs := false
	for i := range x {
		if math.Abs(float64(x[i]-adj[i])) > 1e-4 {
			differs = true
			break
		}
	}
	if !differs {
		t.Fatal("rotate_half matched adjacent NeoX — test setup wrong")
	}
}
