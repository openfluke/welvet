//go:build !amd64

package simd

import "unsafe"

func dotTileF64Simd(x, w *float64, n int, prev float64) float64 {
	if n <= 0 {
		return prev
	}
	xs := unsafe.Slice(x, n)
	ws := unsafe.Slice(w, n)
	sum := prev
	for i := 0; i < n; i++ {
		sum += xs[i] * ws[i]
	}
	return sum
}
