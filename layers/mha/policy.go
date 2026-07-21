package mha

import "fmt"

// MaskKind selects which query/key pairs may attend.
type MaskKind int

const (
	MaskUnspecified   MaskKind = iota // Validate picks Causal if Causal==true else Causal default
	MaskCausal                        // decoder LM
	MaskBidirectional                 // encoder / diffusion self-attn
	MaskSlidingWindow                 // local attention (causal or bidirectional via WindowCausal)
	MaskPrefixLM                      // bidirectional prefix, causal suffix (UL2 / PrefixLM)
	MaskCustom                        // Layer.Allow(q,k) required — hard-error if nil
)

func (m MaskKind) String() string {
	switch m {
	case MaskUnspecified:
		return "unspecified"
	case MaskCausal:
		return "causal"
	case MaskBidirectional:
		return "bidirectional"
	case MaskSlidingWindow:
		return "sliding_window"
	case MaskPrefixLM:
		return "prefix_lm"
	case MaskCustom:
		return "custom"
	default:
		return fmt.Sprintf("Mask(%d)", int(m))
	}
}

// PosKind selects positional encoding applied to Q/K (and optionally context K).
type PosKind int

const (
	PosRoPE PosKind = iota // rotary (default for decoder)
	PosNone                // no positional (absolute embeds elsewhere, or ALiBi-only)
	PosALiBi               // attention-linear bias on scores
	PosRoPEALiBi           // RoPE on Q/K + ALiBi on scores
)

func (p PosKind) String() string {
	switch p {
	case PosRoPE:
		return "rope"
	case PosNone:
		return "none"
	case PosALiBi:
		return "alibi"
	case PosRoPEALiBi:
		return "rope+alibi"
	default:
		return fmt.Sprintf("Pos(%d)", int(p))
	}
}

// AttnMode is self- vs cross-attention.
type AttnMode int

const (
	ModeSelf  AttnMode = iota // Q/K/V from the same sequence (default)
	ModeCross                 // Q from input; K/V from Context (encoder / diffusion cond)
)

func (m AttnMode) String() string {
	switch m {
	case ModeSelf:
		return "self"
	case ModeCross:
		return "cross"
	default:
		return fmt.Sprintf("Mode(%d)", int(m))
	}
}

// SoftmaxKind is the attention normalization.
type SoftmaxKind int

const (
	SoftmaxStandard SoftmaxKind = iota // max-subtract softmax (default)
	SoftmaxSigmoid                     // independent σ(score); diffusion / linear hybrids
)

func (s SoftmaxKind) String() string {
	switch s {
	case SoftmaxStandard:
		return "softmax"
	case SoftmaxSigmoid:
		return "sigmoid"
	default:
		return fmt.Sprintf("Softmax(%d)", int(s))
	}
}

// Role is a documentation / preset tag for how this block is used in a larger model.
// It does not change math by itself — Mask/Pos/Mode do — but keeps transformer /
// diffusion / encoder graphs self-describing for ENTITY + tooling.
type Role int

const (
	RoleGeneric Role = iota
	RoleDecoderSelf
	RoleEncoderSelf
	RoleCrossAttn
	RoleDiffusionSelf
	RoleDiffusionCross
	RolePrefixLM
)

func (r Role) String() string {
	switch r {
	case RoleGeneric:
		return "generic"
	case RoleDecoderSelf:
		return "decoder_self"
	case RoleEncoderSelf:
		return "encoder_self"
	case RoleCrossAttn:
		return "cross"
	case RoleDiffusionSelf:
		return "diffusion_self"
	case RoleDiffusionCross:
		return "diffusion_cross"
	case RolePrefixLM:
		return "prefix_lm"
	default:
		return fmt.Sprintf("Role(%d)", int(r))
	}
}
