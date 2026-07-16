package mha

import "math"

// attnForward writes attnOut [seq*qDim] for one batch row under cfg.Mask / ALiBi.
//
// Self-attn: kvLen is the number of valid KV positions (typically seqBase+seqLen for
// causal decode cache, or seqLen for a fresh bidirectional pass starting at 0).
// Cross-attn: seqBase is usually 0; kvLen = context length; Q is query seq only.
//
// scoresScratch is optional reused buffer (grown as needed) — avoids per-head allocs
// on the causal decode path (Lucy-style).
func attnForward(
	cfg Config,
	attnOut, Q, cacheK, cacheV []float64,
	seqLen, seqBase, kvLen, msl, qDim, kvDim int,
	allow func(qPos, kPos int) bool,
	scoresScratch *[]float64,
) {
	numHeads, numKVHeads, headDim := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	headsPerKV := numHeads / numKVHeads
	scale := cfg.Scale()
	if allow == nil {
		allow = func(q, k int) bool { return Allow(cfg, q, k) }
	}
	causalFast := cfg.Mode == ModeSelf && cfg.Mask == MaskCausal && cfg.Pos != PosALiBi

	for s := 0; s < seqLen; s++ {
		qPos := seqBase + s
		for h := 0; h < numHeads; h++ {
			kvHead := h / headsPerKV
			qOff := s*qDim + h*headDim
			aOff := s*qDim + h*headDim

			if causalFast {
				need := qPos + 1
				scores := attnScoresBuf(scoresScratch, need)
				maxScore := float64(-1e9)
				for kPos := 0; kPos <= qPos; kPos++ {
					kIdx := kPos % msl
					var dot float64
					kBase := kIdx*kvDim + kvHead*headDim
					for d := 0; d < headDim; d++ {
						dot += Q[qOff+d] * cacheK[kBase+d]
					}
					score := dot * scale
					scores[kPos] = score
					if score > maxScore {
						maxScore = score
					}
				}
				var expSum float64
				for kPos := 0; kPos <= qPos; kPos++ {
					scores[kPos] = math.Exp(scores[kPos] - maxScore)
					expSum += scores[kPos]
				}
				for d := 0; d < headDim; d++ {
					var sum float64
					for kPos := 0; kPos <= qPos; kPos++ {
						sum += scores[kPos] * cacheV[(kPos%msl)*kvDim+kvHead*headDim+d]
					}
					attnOut[aOff+d] = sum / expSum
				}
				continue
			}

			scores := attnScoresBuf(scoresScratch, kvLen)[:0]
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
				scores = append(scores, score)
				kPositions = append(kPositions, kPos)
				if score > maxScore {
					maxScore = score
				}
			}
			if len(scores) == 0 {
				for d := 0; d < headDim; d++ {
					attnOut[aOff+d] = 0
				}
				continue
			}
			var expSum float64
			for i := range scores {
				scores[i] = math.Exp(scores[i] - maxScore)
				expSum += scores[i]
			}
			for d := 0; d < headDim; d++ {
				var sum float64
				for i, kPos := range kPositions {
					vIdx := kPos % msl
					sum += scores[i] * cacheV[vIdx*kvDim+kvHead*headDim+d]
				}
				attnOut[aOff+d] = sum / expSum
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
func attnBackward(
	cfg Config,
	gradPre, Q, cacheK, cacheV, gQ, gK, gV []float64,
	seqLen, seqBase, kvLen, msl, qDim, kvDim int,
	allow func(qPos, kPos int) bool,
) {
	numHeads, numKVHeads, headDim := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	headsPerKV := numHeads / numKVHeads
	scale := cfg.Scale()
	if allow == nil {
		allow = func(q, k int) bool { return Allow(cfg, q, k) }
	}
	for h := 0; h < numHeads; h++ {
		kvHead := h / headsPerKV
		for qPosRel := 0; qPosRel < seqLen; qPosRel++ {
			qPos := seqBase + qPosRel
			kPositions := make([]int, 0, kvLen)
			scores := make([]float64, 0, kvLen)
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
				scores = append(scores, score)
				kPositions = append(kPositions, kPos)
				if score > maxScore {
					maxScore = score
				}
			}
			if len(scores) == 0 {
				continue
			}
			var expSum float64
			for i := range scores {
				scores[i] = math.Exp(scores[i] - maxScore)
				expSum += scores[i]
			}
			for i := range scores {
				scores[i] /= expSum
			}
			for d := 0; d < headDim; d++ {
				dy := gradPre[qPosRel*qDim+h*headDim+d]
				var dSSum float64
				for i, kPos := range kPositions {
					vIdx := kPos % msl
					gV[vIdx*kvDim+kvHead*headDim+d] += scores[i] * dy
					dSSum += cacheV[vIdx*kvDim+kvHead*headDim+d] * dy * scores[i]
				}
				for i, kPos := range kPositions {
					kIdx := kPos % msl
					dScore := (scores[i]*dy*cacheV[kIdx*kvDim+kvHead*headDim+d] - scores[i]*dSSum) * scale
					gQ[qPosRel*qDim+h*headDim+d] += dScore * cacheK[kIdx*kvDim+kvHead*headDim+d]
					gK[kIdx*kvDim+kvHead*headDim+d] += dScore * Q[qPosRel*qDim+h*headDim+d]
				}
			}
		}
	}
}
