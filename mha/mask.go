package mha

import "math"

// Allow reports whether query position qPos may attend to key position kPos.
// Positions are absolute (including KV cache offset). For cross-attn, kPos indexes
// the context sequence (0..ctxLen-1) and qPos indexes the query sequence.
func Allow(cfg Config, qPos, kPos int) bool {
	switch cfg.Mask {
	case MaskBidirectional:
		return true
	case MaskCausal:
		return kPos <= qPos
	case MaskSlidingWindow:
		w := cfg.windowWidth()
		if cfg.WindowCausal || cfg.Causal {
			return kPos <= qPos && kPos >= qPos-w+1
		}
		d := qPos - kPos
		if d < 0 {
			d = -d
		}
		return d < w
	case MaskPrefixLM:
		pref := cfg.PrefixLen
		if qPos < pref && kPos < pref {
			return true
		}
		return kPos <= qPos
	case MaskCustom:
		return false
	default:
		return kPos <= qPos
	}
}

func alibiBias(cfg Config, head, qPos, kPos int) float64 {
	if !cfg.UsesALiBi() {
		return 0
	}
	slope := alibiSlope(cfg, head)
	d := math.Abs(float64(qPos - kPos))
	return -slope * d
}

func alibiSlope(cfg Config, head int) float64 {
	if head >= 0 && head < len(cfg.ALiBiSlopes) {
		return cfg.ALiBiSlopes[head]
	}
	n := cfg.NumHeads
	if n <= 0 {
		n = 1
	}
	return math.Pow(2, -float64(head+1)*8.0/float64(n))
}
