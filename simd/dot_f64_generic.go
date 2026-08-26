//go:build !amd64

package simd

func dotTileF64Simd(x, w *float64, n int, prev float64) float64 {
	sum := prev
	for i := 0; i < n; i++ {
		sum += x[i] * w[i]
	}
	return sum
}
