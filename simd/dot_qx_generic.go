//go:build !amd64

package simd

func dotQ8_0RowSimd(in []float32, scales []float32, qs []int8, baseW, n int, prev float64) float64 {
	return dotQ8_0RowGo(in, scales, qs, baseW, n, prev)
}

func dotQ8_0Rows4Simd(in []float32, scales []float32, qs []int8, baseW, n int, out []float32) {
	for r := 0; r < 4; r++ {
		out[r] = float32(dotQ8_0RowGo(in, scales, qs, baseW+r*n, n, 0))
	}
}

func dotQ4_1RowSimd(in []float32, scales, mins []float32, packed []uint32, baseW, n int, prev float64) float64 {
	return dotQ4_1RowGo(in, scales, mins, packed, baseW, n, prev)
}

func dotQ4_1Rows4Simd(in []float32, scales, mins []float32, packed []uint32, baseW, n int, out []float32) {
	for r := 0; r < 4; r++ {
		out[r] = float32(dotQ4_1RowGo(in, scales, mins, packed, baseW+r*n, n, 0))
	}
}
