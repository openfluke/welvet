package qwentts

import (
	"math"
	"math/rand"
	"sort"
)

// sampler holds RNG + sampling hyperparameters for the talker (main) and the
// code predictor (sub-talker).
type sampler struct {
	rng *rand.Rand

	doSample bool
	// main (code0 / talker)
	temp    float32
	topK    int
	topP    float32
	repPen  float32
	// sub-talker (code predictor)
	cpDoSample bool
	cpTemp     float32
	cpTopK     int
	cpTopP     float32
}

func newSampler(seed int64, opts SpeakOpts) *sampler {
	return &sampler{
		rng:        rand.New(rand.NewSource(seed)),
		doSample:   opts.DoSample,
		temp:       0.9,
		topK:       50,
		topP:       1.0,
		repPen:     1.05,
		cpDoSample: opts.DoSample,
		cpTemp:     0.9,
		cpTopK:     50,
		cpTopP:     1.0,
	}
}

// sampleCodec samples code0 from talker logits with repetition penalty.
func (s *sampler) sampleCodec(logits []float32, prev []int) int {
	if !s.doSample || s.temp <= 0 {
		return argmax(logits)
	}
	work := make([]float32, len(logits))
	copy(work, logits)
	if s.repPen != 1.0 {
		for _, t := range prev {
			if t >= 0 && t < len(work) {
				if work[t] > 0 {
					work[t] /= s.repPen
				} else {
					work[t] *= s.repPen
				}
			}
		}
	}
	return sampleFrom(work, s.temp, s.topK, s.topP, s.rng)
}

// sampleCP samples a codebook index from code-predictor logits.
func (s *sampler) sampleCP(logits []float32) int {
	if !s.cpDoSample || s.cpTemp <= 0 {
		return argmax(logits)
	}
	return sampleFrom(logits, s.cpTemp, s.cpTopK, s.cpTopP, s.rng)
}

func argmax(x []float32) int {
	best := 0
	bv := float32(math.Inf(-1))
	for i, v := range x {
		if v > bv {
			bv = v
			best = i
		}
	}
	return best
}

type idScore struct {
	id int
	v  float32
}

func sampleFrom(logits []float32, temp float32, topK int, topP float32, rng *rand.Rand) int {
	n := len(logits)
	cand := make([]idScore, n)
	for i := 0; i < n; i++ {
		cand[i] = idScore{i, logits[i] / temp}
	}
	sort.Slice(cand, func(a, b int) bool { return cand[a].v > cand[b].v })
	if topK > 0 && topK < len(cand) {
		cand = cand[:topK]
	}
	// softmax over candidates
	maxV := cand[0].v
	var sum float64
	for i := range cand {
		e := math.Exp(float64(cand[i].v - maxV))
		cand[i].v = float32(e)
		sum += e
	}
	inv := float32(1.0 / sum)
	for i := range cand {
		cand[i].v *= inv
	}
	// top-p (nucleus)
	if topP > 0 && topP < 1.0 {
		var cum float32
		cut := len(cand)
		for i := range cand {
			cum += cand[i].v
			if cum >= topP {
				cut = i + 1
				break
			}
		}
		cand = cand[:cut]
	}
	// renormalize + multinomial
	var total float32
	for i := range cand {
		total += cand[i].v
	}
	r := rng.Float32() * total
	var acc float32
	for i := range cand {
		acc += cand[i].v
		if r <= acc {
			return cand[i].id
		}
	}
	return cand[len(cand)-1].id
}
