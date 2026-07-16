//go:build !amd64 && !arm64

package simd

func dotF16PackedSimd(x []float32, w []byte, i0, n int, prev float64) float64 {
	return dotF16PackedGo(x, w, i0, n, prev)
}

func dotFP8PackedSimd(x []float32, w []byte, i0, n, kind int, prev float64) float64 {
	return dotFP8PackedGo(x, w, i0, n, kind, prev)
}
