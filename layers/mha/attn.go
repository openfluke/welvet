package mha

import (
	"math"
	"math/rand"
)

// attnForward writes attnOut [seq*qDim] for one batch row under cfg.Mask / ALiBi.
//
// SoftmaxStandard: max-subtract softmax over allowed keys.
// SoftmaxSigmoid: independent σ(score) per allowed key (no normalize).
//
// When train && Dropout∈(0,1), applies inverted dropout on attention weights and
// records keep/drop into dropMask (byte 1 = kept). Layout: [seq][head][kPos].
func attnForward(
	cfg Config,
	attnOut, Q, cacheK, cacheV []float64,
	seqLen, seqBase, kvLen, msl, qDim, kvDim int,
	allow func(qPos, kPos int) bool,
	scoresScratch *[]float64,
	train bool,
	dropMask []byte,
	rng *rand.Rand,
) {
	numHeads, numKVHeads, headDim := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	headsPerKV := numHeads / numKVHeads
	scale := cfg.Scale()
	if allow == nil {
		allow = func(q, k int) bool { return Allow(cfg, q, k) }
	}
	useSigmoid := cfg.Softmax == SoftmaxSigmoid
	dropP := cfg.Dropout
	doDrop := train && dropP > 0 && dropP < 1 && dropMask != nil && rng != nil
	keep := 1.0
	if doDrop {
		keep = 1.0 / (1.0 - dropP)
	}
	maskStride := numHeads * kvLen

	for s := 0; s < seqLen; s++ {
		qPos := seqBase + s
		for h := 0; h < numHeads; h++ {
			kvHead := h / headsPerKV
			qOff := s*qDim + h*headDim
			aOff := s*qDim + h*headDim

			n := 0
			scores := attnScoresBuf(scoresScratch, kvLen)
			kPositions := make([]int, 0, kvLen)
			maxScore := float64(-1e9)
			for kPos := 0; kPos < kvLen; kPos++ {
				if !allow(qPos, kPos) {
					continue
				}
				kIdx := kPos % msl
				var dot float64
				kBase := kIdx*kvDim + kvHead*headDim
				for d := 0; d < headDim; d++ {
					dot += Q[qOff+d] * cacheK[kBase+d]
				}
				score := dot*scale + alibiBias(cfg, h, qPos, kPos)
				scores[n] = score
				kPositions = append(kPositions, kPos)
				if score > maxScore {
					maxScore = score
				}
				n++
			}
			if n == 0 {
				for d := 0; d < headDim; d++ {
					attnOut[aOff+d] = 0
				}
				continue
			}
			scores = scores[:n]

			if useSigmoid {
				for i := 0; i < n; i++ {
					scores[i] = 1.0 / (1.0 + math.Exp(-scores[i]))
				}
			} else {
				var expSum float64
				for i := 0; i < n; i++ {
					scores[i] = math.Exp(scores[i] - maxScore)
					expSum += scores[i]
				}
				inv := 1.0 / expSum
				for i := 0; i < n; i++ {
					scores[i] *= inv
				}
			}

			if doDrop {
				base := s*maskStride + h*kvLen
				for i, kPos := range kPositions {
					mi := base + kPos
					if mi >= len(dropMask) {
						continue
					}
					if rng.Float64() < dropP {
						dropMask[mi] = 0
						scores[i] = 0
					} else {
						dropMask[mi] = 1
						scores[i] *= keep
					}
				}
			}

			for d := 0; d < headDim; d++ {
				var sum float64
				for i, kPos := range kPositions {
					vIdx := kPos % msl
					sum += scores[i] * cacheV[vIdx*kvDim+kvHead*headDim+d]
				}
				attnOut[aOff+d] = sum
			}
		}
	}
}

func attnScoresBuf(scratch *[]float64, n int) []float64 {
	if scratch == nil {
		return make([]float64, n)
	}
	if cap(*scratch) < n {
		*scratch = make([]float64, n)
	}
	*scratch = (*scratch)[:n]
	return *scratch
}

// attnBackward accumulates into gQ [seq*qDim], gK/gV [msl*kvDim].
// dropMask/train must match the forward pass when Dropout>0.
func attnBackward(
	cfg Config,
	gradPre, Q, cacheK, cacheV, gQ, gK, gV []float64,
	seqLen, seqBase, kvLen, msl, qDim, kvDim int,
	allow func(qPos, kPos int) bool,
	train bool,
	dropMask []byte,
) {
	numHeads, numKVHeads, headDim := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	headsPerKV := numHeads / numKVHeads
	scale := cfg.Scale()
	if allow == nil {
		allow = func(q, k int) bool { return Allow(cfg, q, k) }
	}
	useSigmoid := cfg.Softmax == SoftmaxSigmoid
	dropP := cfg.Dropout
	doDrop := train && dropP > 0 && dropP < 1 && dropMask != nil
	keep := 1.0
	if doDrop {
		keep = 1.0 / (1.0 - dropP)
	}
	maskStride := numHeads * kvLen

	for h := 0; h < numHeads; h++ {
		kvHead := h / headsPerKV
		for qPosRel := 0; qPosRel < seqLen; qPosRel++ {
			qPos := seqBase + qPosRel
			kPositions := make([]int, 0, kvLen)
			rawScores := make([]float64, 0, kvLen)
			maxScore := float64(-1e9)
			qOff := qPosRel*qDim + h*headDim
			for kPos := 0; kPos < kvLen; kPos++ {
				if !allow(qPos, kPos) {
					continue
				}
				kIdx := kPos % msl
				var dot float64
				kBase := kIdx*kvDim + kvHead*headDim
				for d := 0; d < headDim; d++ {
					dot += Q[qOff+d] * cacheK[kBase+d]
				}
				score := dot*scale + alibiBias(cfg, h, qPos, kPos)
				rawScores = append(rawScores, score)
				kPositions = append(kPositions, kPos)
				if score > maxScore {
					maxScore = score
				}
			}
			n := len(rawScores)
			if n == 0 {
				continue
			}

			baseW := make([]float64, n) // pre-dropout weights
			if useSigmoid {
				for i := 0; i < n; i++ {
					baseW[i] = 1.0 / (1.0 + math.Exp(-rawScores[i]))
				}
			} else {
				var expSum float64
				for i := 0; i < n; i++ {
					baseW[i] = math.Exp(rawScores[i] - maxScore)
					expSum += baseW[i]
				}
				inv := 1.0 / expSum
				for i := 0; i < n; i++ {
					baseW[i] *= inv
				}
			}

			w := make([]float64, n) // post-dropout (matches forward)
			copy(w, baseW)
			if doDrop {
				base := qPosRel*maskStride + h*kvLen
				for i, kPos := range kPositions {
					mi := base + kPos
					if mi < len(dropMask) && dropMask[mi] == 0 {
						w[i] = 0
					} else {
						w[i] *= keep
					}
				}
			}

			gOff := qPosRel*qDim + h*headDim
			dW := make([]float64, n) // ∂L/∂w_post
			for d := 0; d < headDim; d++ {
				dy := gradPre[gOff+d]
				for i, kPos := range kPositions {
					vIdx := kPos % msl
					v := cacheV[vIdx*kvDim+kvHead*headDim+d]
					gV[vIdx*kvDim+kvHead*headDim+d] += w[i] * dy
					dW[i] += v * dy
				}
			}

			// ∂L/∂baseW through inverted dropout
			dBase := make([]float64, n)
			if doDrop {
				base := qPosRel*maskStride + h*kvLen
				for i, kPos := range kPositions {
					mi := base + kPos
					if mi < len(dropMask) && dropMask[mi] == 0 {
						dBase[i] = 0
					} else {
						dBase[i] = dW[i] * keep
					}
				}
			} else {
				copy(dBase, dW)
			}

			dScore := make([]float64, n)
			if useSigmoid {
				for i := 0; i < n; i++ {
					sig := baseW[i]
					dScore[i] = dBase[i] * sig * (1 - sig) * scale
				}
			} else {
				var sum float64
				for i := 0; i < n; i++ {
					sum += dBase[i] * baseW[i]
				}
				for i := 0; i < n; i++ {
					dScore[i] = baseW[i] * (dBase[i] - sum) * scale
				}
			}

			for i, kPos := range kPositions {
				kIdx := kPos % msl
				kBase := kIdx*kvDim + kvHead*headDim
				ds := dScore[i]
				for d := 0; d < headDim; d++ {
					gQ[gOff+d] += ds * cacheK[kBase+d]
					gK[kBase+d] += ds * Q[qOff+d]
				}
			}
		}
	}
}
