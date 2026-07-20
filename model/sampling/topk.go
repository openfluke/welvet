package sampling

import (
	"math"
	"math/rand"
	"sort"
)

// SampleTopK performs top-K sampling with temperature.
// When topK == 1, temperature <= 0, or deterministic is true, returns ArgMax.
func SampleTopK(logits []float32, topK int, temperature float32, deterministic bool) int {
	if len(logits) == 0 {
		return 0
	}
	if topK == 1 || temperature <= 0 || deterministic {
		return ArgMax(logits)
	}

	type pair struct {
		idx int
		val float32
	}
	cands := make([]pair, 0, len(logits))
	for i, v := range logits {
		cands = append(cands, pair{i, v / temperature})
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].val > cands[j].val })
	if topK > 0 && topK < len(cands) {
		cands = cands[:topK]
	}

	maxV := cands[0].val
	var sum float64
	probs := make([]float64, len(cands))
	for i := range cands {
		p := math.Exp(float64(cands[i].val - maxV))
		probs[i] = p
		sum += p
	}

	r := rand.Float64() * sum
	acc := 0.0
	for i := range probs {
		acc += probs[i]
		if r <= acc {
			return cands[i].idx
		}
	}
	return cands[len(cands)-1].idx
}
