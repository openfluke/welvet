package softmax

import (
	"math"
)

// Softmax is stable softmax over logits (float32).
func Softmax(logits []float32) []float32 {
	if len(logits) == 0 {
		return nil
	}
	maxLogit := logits[0]
	for _, v := range logits {
		if v > maxLogit {
			maxLogit = v
		}
	}
	out := make([]float32, len(logits))
	var sumExp float32
	for i, v := range logits {
		e := float32(math.Exp(float64(v - maxLogit)))
		out[i] = e
		sumExp += e
	}
	for i := range out {
		out[i] /= sumExp
	}
	return out
}

// SoftmaxBackward is the softmax Jacobian × upstream grad.
func SoftmaxBackward(gradOutput, softmaxOutput []float32) []float32 {
	n := len(gradOutput)
	gradLogits := make([]float32, n)
	var dotProd float32
	for i := 0; i < n; i++ {
		dotProd += gradOutput[i] * softmaxOutput[i]
	}
	for j := 0; j < n; j++ {
		gradLogits[j] = softmaxOutput[j] * (gradOutput[j] - dotProd)
	}
	return gradLogits
}

// SoftmaxSparseHelper implements sparsemax.
func SoftmaxSparseHelper(logits []float32) []float32 {
	n := len(logits)
	sorted := make([]float32, n)
	copy(sorted, logits)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if sorted[i] < sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}
	cumSum := float32(0)
	k := 0
	for i := 0; i < n; i++ {
		cumSum += sorted[i]
		if sorted[i]-(cumSum-1.0)/float32(i+1) > 0 {
			k = i + 1
		} else {
			break
		}
	}
	tau := float32(0)
	if k > 0 {
		cumSum = 0
		for i := 0; i < k; i++ {
			cumSum += sorted[i]
		}
		tau = (cumSum - 1.0) / float32(k)
	}
	result := make([]float32, n)
	for i := 0; i < n; i++ {
		result[i] = float32(math.Max(0, float64(logits[i]-tau)))
	}
	return result
}

// SoftmaxEntmaxHelper implements entmax-α (α≈1.5 default).
func SoftmaxEntmaxHelper(logits []float32, alpha float32) []float32 {
	if alpha <= 1.0 {
		return Softmax(logits)
	}
	if alpha >= 2.0 {
		return SoftmaxSparseHelper(logits)
	}
	weight := alpha - 1.0
	s1 := Softmax(logits)
	s2 := SoftmaxSparseHelper(logits)
	res := make([]float32, len(logits))
	var sum float32
	for i := range res {
		res[i] = (1-weight)*s1[i] + weight*s2[i]
		sum += res[i]
	}
	for i := range res {
		res[i] /= sum
	}
	return res
}
