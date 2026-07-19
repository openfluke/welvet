// Package sampling provides token selection helpers (greedy ArgMax for v0).
package sampling

// ArgMax returns the index of the maximum logit (greedy).
func ArgMax(logits []float32) int {
	if len(logits) == 0 {
		return 0
	}
	maxIdx := 0
	maxVal := logits[0]
	for i, v := range logits {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}
