package mha

import "math"

// attnForward writes attnOut [seq*qDim] for one batch row under cfg.Mask / ALiBi.
//
// Self-attn: kvLen is the number of valid KV positions (typically seqBase+seqLen for
// causal decode cache, or seqLen for a fresh bidirectional pass starting at 0).
// Cross-attn: seqBase is usually 0; kvLen = context length; Q is query seq only.
func attnForward(
	cfg Config,
	attnOut, Q, cacheK, cacheV []float64,
	seqLen, seqBase, kvLen, msl, qDim, kvDim int,
	allow func(qPos, kPos int) bool,
) {
	numHeads, numKVHeads, headDim := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	headsPerKV := numHeads / numKVHeads
	scale := cfg.Scale()
	if allow == nil {
		allow = func(q, k int) bool { return Allow(cfg, q, k) }
	}
	for s := 0; s < seqLen; s++ {
		qPos := seqBase + s
		for h := 0; h < numHeads; h++ {
			kvHead := h / headsPerKV
			scores := make([]float64, 0, kvLen)
			kPositions := make([]int, 0, kvLen)
			maxScore := float64(-1e9)
			qOff := s*qDim + h*headDim
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
				aOff := s*qDim + h*headDim
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
			aOff := s*qDim + h*headDim
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
