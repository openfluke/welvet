package mha

// Preset constructors — transformers, diffusion, PrefixLM, local-attn.
// Math is driven by Mask/Pos/Mode; Role is for tooling / ENTITY labels.

// DecoderCausal is standard autoregressive self-attn (GPT / Llama style).
func DecoderCausal(dModel, numHeads, numKVHeads int) Config {
	return Config{
		DModel: dModel, NumHeads: numHeads, NumKVHeads: numKVHeads,
		Mask: MaskCausal, Pos: PosRoPE, Mode: ModeSelf, Causal: true,
		Role: RoleDecoderSelf,
	}
}

// EncoderBidirectional is BERT / encoder-tower self-attn.
func EncoderBidirectional(dModel, numHeads int) Config {
	return Config{
		DModel: dModel, NumHeads: numHeads,
		Mask: MaskBidirectional, Pos: PosNone, Mode: ModeSelf,
		Role: RoleEncoderSelf,
	}
}

// CrossAttention is encoder–decoder or diffusion conditioning cross-attn.
// Caller must SetContext / ForwardWithContext with encoder (or cond) states.
func CrossAttention(dModel, numHeads, numKVHeads int) Config {
	return Config{
		DModel: dModel, NumHeads: numHeads, NumKVHeads: numKVHeads,
		Mask: MaskBidirectional, Pos: PosNone, Mode: ModeCross,
		Role: RoleCrossAttn,
	}
}

// DiffusionSelf is UNet / DiT self-attention over spatial or token grid (full mask).
func DiffusionSelf(dModel, numHeads int) Config {
	return Config{
		DModel: dModel, NumHeads: numHeads,
		Mask: MaskBidirectional, Pos: PosNone, Mode: ModeSelf,
		Role: RoleDiffusionSelf,
	}
}

// DiffusionCross is conditioning cross-attn inside a diffusion UNet / DiT block.
func DiffusionCross(dModel, numHeads, numKVHeads int) Config {
	cfg := CrossAttention(dModel, numHeads, numKVHeads)
	cfg.Role = RoleDiffusionCross
	return cfg
}

// PrefixLMAttn is UL2 / Prefix-LM style (bidirectional prefix, causal suffix).
func PrefixLMAttn(dModel, numHeads, prefixLen int) Config {
	return Config{
		DModel: dModel, NumHeads: numHeads, PrefixLen: prefixLen,
		Mask: MaskPrefixLM, Pos: PosRoPE, Mode: ModeSelf, Causal: true,
		Role: RolePrefixLM,
	}
}

// LocalCausal is sliding-window causal attention (Longformer / Mistral-style local).
func LocalCausal(dModel, numHeads, window int) Config {
	return Config{
		DModel: dModel, NumHeads: numHeads, Window: window, WindowCausal: true,
		Mask: MaskSlidingWindow, Pos: PosRoPE, Mode: ModeSelf, Causal: true,
		Role: RoleDecoderSelf,
	}
}

// ALiBiCausal is causal attention with ALiBi (no RoPE).
func ALiBiCausal(dModel, numHeads int) Config {
	return Config{
		DModel: dModel, NumHeads: numHeads,
		Mask: MaskCausal, Pos: PosALiBi, Mode: ModeSelf, Causal: true,
		Role: RoleDecoderSelf,
	}
}
