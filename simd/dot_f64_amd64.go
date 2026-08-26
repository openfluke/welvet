//go:build amd64

package simd

//go:noescape
func dotF64AccF64Avx2(x, w *float64, n int, prev float64) float64

func dotTileF64Simd(x, w *float64, n int, prev float64) float64 {
	if n <= 0 {
		return prev
	}
	return dotF64AccF64Avx2(x, w, n, prev)
}
