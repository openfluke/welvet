package hf

import (
	"fmt"
	"strings"
)

// EffectiveConfig returns the decoder config map. Nested multimodal configs
// (Qwen3.5 / Bonsai) keep text hyperparams under "text_config".
func EffectiveConfig(config map[string]any) map[string]any {
	if config == nil {
		return nil
	}
	if tc, ok := config["text_config"].(map[string]any); ok && tc != nil {
		// Prefer text_config ints when top-level lacks them.
		out := make(map[string]any, len(tc)+8)
		for k, v := range config {
			out[k] = v
		}
		for k, v := range tc {
			out[k] = v
		}
		return out
	}
	return config
}

// LayerTypes returns per-layer mixer kinds ("linear_attention" / "full_attention").
// Empty means uniform full attention (Llama-style).
func LayerTypes(config map[string]any) []string {
	cfg := EffectiveConfig(config)
	raw, ok := cfg["layer_types"].([]any)
	if !ok || len(raw) == 0 {
		return nil
	}
	out := make([]string, 0, len(raw))
	for _, v := range raw {
		s, _ := v.(string)
		out = append(out, s)
	}
	return out
}

// IsQwen35Hybrid reports Qwen3.5 / Bonsai hybrid (Gated DeltaNet + full attn).
// Dense Qwen3 (e.g. Bonsai-8B) often ships layer_types of all "full_attention" —
// that alone must not count as hybrid.
func IsQwen35Hybrid(config map[string]any) bool {
	mt := strings.ToLower(ConfigStringDefault(config, "model_type", ""))
	if mt == "qwen3_5" || mt == "qwen3_5_text" {
		return true
	}
	for _, a := range architectureStrings(config) {
		al := strings.ToLower(a)
		if strings.Contains(al, "qwen3_5") || strings.Contains(al, "qwen3.5") {
			return true
		}
	}
	hasLinear := false
	for _, t := range LayerTypes(config) {
		if t == "linear_attention" {
			hasLinear = true
			break
		}
	}
	return hasLinear && strings.Contains(mt, "qwen3")
}

// LinearAttnDims holds Gated DeltaNet shape knobs from text_config.
type LinearAttnDims struct {
	ConvKernel   int
	NumKeyHeads  int
	NumValueHeads int
	KeyHeadDim   int
	ValueHeadDim int
}

// ParseLinearAttnDims reads linear_* fields (zero if absent).
func ParseLinearAttnDims(config map[string]any) LinearAttnDims {
	cfg := EffectiveConfig(config)
	return LinearAttnDims{
		ConvKernel:    ConfigIntDefault(cfg, "linear_conv_kernel_dim", 4),
		NumKeyHeads:   ConfigIntDefault(cfg, "linear_num_key_heads", 0),
		NumValueHeads: ConfigIntDefault(cfg, "linear_num_value_heads", 0),
		KeyHeadDim:    ConfigIntDefault(cfg, "linear_key_head_dim", 0),
		ValueHeadDim:  ConfigIntDefault(cfg, "linear_value_head_dim", 0),
	}
}

// QuantBitsGroup returns MLX/HF quantization bits + group_size when present.
func QuantBitsGroup(config map[string]any) (bits, group int, ok bool) {
	for _, root := range []map[string]any{config, EffectiveConfig(config)} {
		if root == nil {
			continue
		}
		q, _ := root["quantization"].(map[string]any)
		if q == nil {
			continue
		}
		b, bok := ConfigInt(q, "bits")
		g, gok := ConfigInt(q, "group_size")
		if bok && gok && b > 0 && g > 0 {
			return b, g, true
		}
	}
	return 0, 0, false
}

// ValidateHybridDims ensures GDN dims are coherent.
func ValidateHybridDims(lin LinearAttnDims) error {
	if lin.NumKeyHeads <= 0 || lin.NumValueHeads <= 0 || lin.KeyHeadDim <= 0 || lin.ValueHeadDim <= 0 {
		return fmt.Errorf("incomplete linear attention dims")
	}
	if lin.NumValueHeads%lin.NumKeyHeads != 0 {
		return fmt.Errorf("linear_num_value_heads %% linear_num_key_heads != 0")
	}
	return nil
}
