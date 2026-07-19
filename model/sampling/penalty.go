package sampling

import "math"

// ApplyRepetitionPenalty down-weights tokens already seen in the recent window.
// penalty <= 1 leaves logits unchanged. Matches loom/poly chat defaults.
func ApplyRepetitionPenalty(logits []float32, tokens []uint32, penalty float32, window int) {
	if penalty <= 1 || len(tokens) == 0 || len(logits) == 0 {
		return
	}
	if window <= 0 {
		window = 64
	}
	start := len(tokens) - window
	if start < 0 {
		start = 0
	}
	seen := make(map[uint32]struct{}, window)
	for _, tok := range tokens[start:] {
		if int(tok) >= len(logits) {
			continue
		}
		if _, ok := seen[tok]; ok {
			continue
		}
		seen[tok] = struct{}{}
		if logits[tok] > 0 {
			logits[tok] /= penalty
		} else {
			logits[tok] *= penalty
		}
	}
}

// BanIDs sets listed token logits to -Inf so ArgMax cannot pick them.
func BanIDs(logits []float32, ids []int) {
	neg := float32(math.Inf(-1))
	for _, id := range ids {
		if id >= 0 && id < len(logits) {
			logits[id] = neg
		}
	}
}

// HasRepeatedNGram reports whether the last n tokens already appear as the
// immediately preceding n-gram (classic small-model loop).
func HasRepeatedNGram(tokens []uint32, n int) bool {
	if n <= 0 || len(tokens) < 2*n {
		return false
	}
	a := tokens[len(tokens)-2*n : len(tokens)-n]
	b := tokens[len(tokens)-n:]
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// HasRepeatedNGramRecent reports whether the last n tokens appear anywhere in
// the preceding lookback window (catches paragraph loops that aren't adjacent).
func HasRepeatedNGramRecent(tokens []uint32, n, lookback int) bool {
	if n <= 0 || len(tokens) < n+n {
		return false
	}
	if lookback < n {
		lookback = n
	}
	needle := tokens[len(tokens)-n:]
	end := len(tokens) - n
	start := end - lookback
	if start < 0 {
		start = 0
	}
	for i := start; i <= end-n; i++ {
		match := true
		for j := 0; j < n; j++ {
			if tokens[i+j] != needle[j] {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}
