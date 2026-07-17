//go:build amd64

package quant

func binaryWordSum1Impl(x []float32, word uint32) float32 {
	if len(x) < 32 {
		return binaryWordSum1Go(x, word)
	}
	return binaryWordSum1Avx2(&x[0], word)
}

func binaryWordSum1Go(x []float32, word uint32) float32 {
	var sum1 float32
	w := word
	n := 32
	if len(x) < n {
		n = len(x)
	}
	for j := 0; j < n; j++ {
		sum1 += x[j] * float32(w&1)
		w >>= 1
	}
	return sum1
}

//go:noescape
func binaryWordSum1Avx2(x *float32, word uint32) float32
