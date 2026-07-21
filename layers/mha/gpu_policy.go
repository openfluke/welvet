package mha

// GPUAttnSupported reports whether ForwardWebGPU/BackwardWebGPU can run RoPE +
// attention on device (decoder gate).
//
// GPU path: ModeSelf, SoftmaxStandard, no train-time dropout, Pos RoPE or PosNone,
// MaskCausal or MaskBidirectional, GQA via NumKVHeads. QKNorm uses device RMSNorm.
//
// Still host-only: ModeCross, SoftmaxSigmoid, train Dropout>0, ALiBi / RoPE+ALiBi,
// sliding-window / prefix-LM / custom masks.
func GPUAttnSupported(l *Layer) bool {
	return gpuAttnSupported(l)
}

func gpuAttnSupported(l *Layer) bool {
	if l == nil {
		return false
	}
	cfg := l.Cfg
	if cfg.Mode != ModeSelf {
		return false
	}
	if cfg.Softmax != SoftmaxStandard {
		return false
	}
	if l.Training && cfg.Dropout > 0 {
		return false
	}
	switch cfg.Pos {
	case PosRoPE, PosNone:
	default:
		return false
	}
	switch cfg.Mask {
	case MaskCausal, MaskBidirectional:
	default:
		return false
	}
	if cfg.UsesRoPE() && cfg.HeadDim%2 != 0 {
		return false
	}
	if cfg.QKNorm {
		if len(l.QNormWeight) < cfg.HeadDim || len(l.KNormWeight) < cfg.HeadDim {
			return false
		}
	}
	return true
}
