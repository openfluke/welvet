package simd

// DotTileF64 computes sum(x[i]*w[i] for i in [i0,i1)) + prev in float64.
// Used when weight/activation compute wire is float64 (not forced through f32).
func DotTileF64(x, w []float64, i0, i1 int, prev float64) float64 {
	if i0 >= i1 {
		return prev
	}
	sum := prev
	for i := i0; i < i1; i++ {
		sum += x[i] * w[i]
	}
	return sum
}

// SaxpyF64AccF64 computes acc[i] += alpha * x[i] for i in [0,n).
func SaxpyF64AccF64(acc []float64, alpha float64, x []float64, n int) {
	if n <= 0 || len(acc) < n || len(x) < n {
		return
	}
	for i := 0; i < n; i++ {
		acc[i] += alpha * x[i]
	}
}
