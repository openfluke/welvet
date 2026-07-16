package mha

import (
	"fmt"
	"math"
)

// Config is the MHA geometry + attention policy.
//
// Axes that future-proof transformers / diffusion / PrefixLM / local-attn:
//   - Mask (causal, bidirectional, sliding window, prefix-LM, custom)
//   - Pos (RoPE, none, ALiBi, RoPE+ALiBi)
//   - Mode (self vs cross — K/V from Context)
//   - Softmax, QK-RMSNorm, GQA/MQA via NumKVHeads
//
// Causal remains as a convenience alias for MaskCausal (Validate syncs it).
type Config struct {
	DModel     int
	NumHeads   int
	NumKVHeads int // 0 → NumHeads (MHA); 1 → MQA; else GQA
	HeadDim    int // 0 → DModel/NumHeads
	MaxSeqLen  int // KV cache length; 0 → 512

	// Policy
	Mask    MaskKind
	Pos     PosKind
	Mode    AttnMode
	Softmax SoftmaxKind
	Role    Role // descriptive only

	// Causal is legacy: if Mask is zero-value and Causal==true → MaskCausal.
	// Prefer setting Mask explicitly.
	Causal bool

	// Sliding window (MaskSlidingWindow). Tokens outside the window are blocked.
	Window       int  // 0 → MaxSeqLen (full); >0 → local width
	WindowCausal bool // true → causal window (decoder local); false → symmetric

	// PrefixLM: first PrefixLen positions are bidirectional among themselves.
	PrefixLen int

	// RoPE
	RoPETheta     float64 // 0 → 10000
	RoPEOnContext bool    // cross-attn: also rotate context K (rare; default false)

	// ALiBi
	ALiBiSlopes []float64 // per head; empty → geometric default

	// QK RMSNorm (Qwen-style); Eps 0 → 1e-6 when QKNorm
	QKNorm    bool
	QKNormEps float64

	// ScaleOverride > 0 replaces 1/sqrt(HeadDim)
	ScaleOverride float64

	// Dropout reserved (must be 0 until train-time dropout lands)
	Dropout float64
}

// Validate fills defaults and rejects unsupported combinations loudly.
func (c *Config) Validate() error {
	if c == nil {
		return fmt.Errorf("mha: nil config")
	}
	if c.DModel <= 0 || c.NumHeads <= 0 {
		return fmt.Errorf("mha: need DModel>0 and NumHeads>0")
	}
	if c.HeadDim <= 0 {
		if c.DModel%c.NumHeads != 0 {
			return fmt.Errorf("mha: HeadDim unset and DModel %d not divisible by NumHeads %d", c.DModel, c.NumHeads)
		}
		c.HeadDim = c.DModel / c.NumHeads
	}
	if c.NumKVHeads <= 0 {
		c.NumKVHeads = c.NumHeads
	}
	if c.NumHeads%c.NumKVHeads != 0 {
		return fmt.Errorf("mha: NumHeads %d not divisible by NumKVHeads %d", c.NumHeads, c.NumKVHeads)
	}
	if c.MaxSeqLen <= 0 {
		c.MaxSeqLen = 512
	}

	// Sync legacy Causal ↔ Mask
	if c.Mask == MaskUnspecified {
		if c.Causal {
			c.Mask = MaskCausal
		} else {
			// Default for new configs that set nothing: causal decoder (loom parity).
			c.Mask = MaskCausal
			c.Causal = true
		}
	}
	if c.Mask == MaskCausal {
		c.Causal = true
	}

	if c.RoPETheta <= 0 {
		c.RoPETheta = 10000
	}
	if c.QKNorm && c.QKNormEps <= 0 {
		c.QKNormEps = 1e-6
	}
	if c.Softmax == SoftmaxSigmoid {
		return fmt.Errorf("mha: SoftmaxSigmoid not implemented yet (hard-error, no silent softmax)")
	}
	if c.Dropout != 0 {
		return fmt.Errorf("mha: Dropout=%v not implemented yet (set 0)", c.Dropout)
	}
	switch c.Mask {
	case MaskCausal, MaskBidirectional, MaskSlidingWindow, MaskPrefixLM, MaskCustom:
	default:
		return fmt.Errorf("mha: unknown Mask %v", c.Mask)
	}
	if c.Mask == MaskSlidingWindow && c.Window < 0 {
		return fmt.Errorf("mha: Window must be >= 0")
	}
	if c.Mask == MaskPrefixLM && c.PrefixLen < 0 {
		return fmt.Errorf("mha: PrefixLen must be >= 0")
	}
	switch c.Pos {
	case PosRoPE, PosNone, PosALiBi, PosRoPEALiBi:
	default:
		return fmt.Errorf("mha: unknown Pos %v", c.Pos)
	}
	switch c.Mode {
	case ModeSelf, ModeCross:
	default:
		return fmt.Errorf("mha: unknown Mode %v", c.Mode)
	}
	if c.Mode == ModeCross && c.Mask == MaskCausal {
		// Cross-attn is usually full over context; allow causal but warn via Role.
	}
	return nil
}

func (c Config) QDim() int       { return c.NumHeads * c.HeadDim }
func (c Config) KVDim() int      { return c.NumKVHeads * c.HeadDim }
func (c Config) HeadsPerKV() int { return c.NumHeads / c.NumKVHeads }

func (c Config) Scale() float64 {
	if c.ScaleOverride > 0 {
		return c.ScaleOverride
	}
	return 1.0 / math.Sqrt(float64(c.HeadDim))
}

func (c Config) windowWidth() int {
	if c.Window > 0 {
		return c.Window
	}
	return c.MaxSeqLen
}

// UsesRoPE reports whether Q/K get rotary.
func (c Config) UsesRoPE() bool {
	return c.Pos == PosRoPE || c.Pos == PosRoPEALiBi
}

// UsesALiBi reports whether scores get ALiBi bias.
func (c Config) UsesALiBi() bool {
	return c.Pos == PosALiBi || c.Pos == PosRoPEALiBi
}
