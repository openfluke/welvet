package simd

// DotBinaryWord computes sum_j scale * (2*bit_j - 1) * x[j] for j in [0,n),
// where bit_j is bit j of word (1 → +scale, 0 → −scale). n must be in 1..32.
// This is the BitNet/binary fused group kernel — no full-row decode.
func DotBinaryWord(x []float32, word uint32, scale float32, n int) float64 {
	if n <= 0 || len(x) < n {
		return 0
	}
	if n > 32 {
		n = 32
	}
	var sum1, sumAll float64
	for j := 0; j < n; j++ {
		v := float64(x[j])
		sumAll += v
		if (word>>uint(j))&1 != 0 {
			sum1 += v
		}
	}
	return float64(scale) * (2*sum1 - sumAll)
}

// DotBinaryWordOffset is DotBinaryWord starting at bit bitOff of word (bitOff+n ≤ 32).
func DotBinaryWordOffset(x []float32, word uint32, scale float32, bitOff, n int) float64 {
	if n <= 0 || len(x) < n || bitOff < 0 || bitOff+n > 32 {
		return 0
	}
	var sum1, sumAll float64
	for j := 0; j < n; j++ {
		v := float64(x[j])
		sumAll += v
		if (word>>uint(bitOff+j))&1 != 0 {
			sum1 += v
		}
	}
	return float64(scale) * (2*sum1 - sumAll)
}
