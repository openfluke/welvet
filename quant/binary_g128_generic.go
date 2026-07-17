//go:build !amd64

package quant

func binaryWordSum1Impl(x []float32, word uint32) float32 {
	return binaryWordSum1Go(x, word)
}

func binaryWordSum1Go(x []float32, word uint32) float32 {
	var sum1 float32
	w := word
	for j := 0; j < 32; j++ {
		sum1 += x[j] * float32(w&1)
		w >>= 1
	}
	return sum1
}
