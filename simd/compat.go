package simd

// Enabled is the Welvet alias for SimdEnabled (Plan 9 AVX2/NEON linked).
func Enabled() bool { return SimdEnabled() }

// DotF32 is a convenience wrapper around DotTile for full-slice FP32 dots.
func DotF32(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if n <= 0 {
		return 0
	}
	return float32(DotTile(a, b, 0, n, 0))
}
