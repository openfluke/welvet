package hf

import (
	"encoding/json"
	"os"
	"strings"
)

// ConfigInt reads an integer from an unmarshaled HF config.json map.
func ConfigInt(config map[string]any, key string) (int, bool) {
	v, ok := config[key]
	if !ok {
		return 0, false
	}
	switch t := v.(type) {
	case float64:
		return int(t), true
	case int:
		return t, true
	case int64:
		return int(t), true
	default:
		return 0, false
	}
}

// ConfigIntDefault returns d when key is missing or not coercible.
func ConfigIntDefault(config map[string]any, key string, d int) int {
	if v, ok := ConfigInt(config, key); ok {
		return v
	}
	return d
}

// ConfigFloat64Default returns d when key is missing or wrong type.
func ConfigFloat64Default(config map[string]any, key string, d float64) float64 {
	v, ok := config[key]
	if !ok {
		return d
	}
	f, ok := v.(float64)
	if !ok {
		return d
	}
	return f
}

// ConfigStringDefault returns d when key is missing.
func ConfigStringDefault(config map[string]any, key string, d string) string {
	if v, ok := config[key].(string); ok {
		return v
	}
	return d
}

// LoadConfigJSON reads and unmarshals config.json.
func LoadConfigJSON(path string) (map[string]any, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var config map[string]any
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, err
	}
	return config, nil
}

// EOSTokenIDs extracts eos_token_id from config (default [2]).
func EOSTokenIDs(config map[string]any) []int {
	var tokens []int
	if eosID, ok := config["eos_token_id"]; ok {
		switch v := eosID.(type) {
		case float64:
			tokens = append(tokens, int(v))
		case []any:
			for _, item := range v {
				if f, ok := item.(float64); ok {
					tokens = append(tokens, int(f))
				}
			}
		}
	}
	if len(tokens) == 0 {
		return []int{2}
	}
	return tokens
}

// ArchitectureKind classifies HF model families Welvet can import.
type ArchitectureKind int

const (
	ArchUnknown ArchitectureKind = iota
	ArchLlamaStyleDecoder
	ArchQwen35Hybrid // Qwen3.5 / Bonsai: Gated DeltaNet + full attention
)

func (k ArchitectureKind) String() string {
	switch k {
	case ArchLlamaStyleDecoder:
		return "llama_style_decoder"
	case ArchQwen35Hybrid:
		return "qwen35_hybrid"
	default:
		return "unknown"
	}
}

// DetectArchitecture returns ArchLlamaStyleDecoder for Llama/Mistral/Qwen/SmolLM/…
// or ArchQwen35Hybrid for Qwen3.5 / Bonsai multimodal text towers.
func DetectArchitecture(config map[string]any) ArchitectureKind {
	if IsQwen35Hybrid(config) {
		return ArchQwen35Hybrid
	}
	cfg := EffectiveConfig(config)
	modelType := strings.ToLower(ConfigStringDefault(cfg, "model_type", ""))
	for _, arch := range architectureStrings(config) {
		a := strings.ToLower(arch)
		switch {
		case strings.Contains(a, "llama"),
			strings.Contains(a, "mistral"),
			strings.Contains(a, "qwen"),
			strings.Contains(a, "smollm"),
			strings.Contains(a, "gemma"),
			strings.Contains(a, "phi"):
			return ArchLlamaStyleDecoder
		}
	}
	switch modelType {
	case "llama", "mistral", "qwen2", "qwen3", "smollm", "gemma", "gemma2", "phi", "phi3":
		return ArchLlamaStyleDecoder
	}
	if strings.Contains(modelType, "smol") || strings.Contains(modelType, "llama") {
		return ArchLlamaStyleDecoder
	}
	if _, ok := ConfigInt(cfg, "num_hidden_layers"); ok {
		if _, ok2 := ConfigInt(cfg, "hidden_size"); ok2 {
			if _, ok3 := ConfigInt(cfg, "num_attention_heads"); ok3 {
				return ArchLlamaStyleDecoder
			}
		}
	}
	return ArchUnknown
}

func architectureStrings(config map[string]any) []string {
	raw, ok := config["architectures"]
	if !ok {
		return nil
	}
	list, ok := raw.([]any)
	if !ok {
		return nil
	}
	out := make([]string, 0, len(list))
	for _, item := range list {
		if s, ok := item.(string); ok {
			out = append(out, s)
		}
	}
	return out
}
