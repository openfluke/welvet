package flux2

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Config holds Flux2Transformer2DModel hyperparameters (Klein 4B defaults).
type Config struct {
	PatchSize                 int     `json:"patch_size"`
	InChannels                int     `json:"in_channels"`
	OutChannels               int     `json:"out_channels"`
	NumLayers                 int     `json:"num_layers"`
	NumSingleLayers           int     `json:"num_single_layers"`
	AttentionHeadDim          int     `json:"attention_head_dim"`
	NumAttentionHeads         int     `json:"num_attention_heads"`
	JointAttentionDim         int     `json:"joint_attention_dim"`
	TimestepGuidanceChannels  int     `json:"timestep_guidance_channels"`
	MLPRatio                  float64 `json:"mlp_ratio"`
	AxesDimsRope              []int   `json:"axes_dims_rope"`
	RopeTheta                 float64 `json:"rope_theta"`
	Eps                       float64 `json:"eps"`
	GuidanceEmbeds            bool    `json:"guidance_embeds"`
}

// InnerDim is num_attention_heads * attention_head_dim.
func (c Config) InnerDim() int { return c.NumAttentionHeads * c.AttentionHeadDim }

// MLPHiddenDim is int(inner_dim * mlp_ratio).
func (c Config) MLPHiddenDim() int { return int(float64(c.InnerDim()) * c.MLPRatio) }

// DefaultConfig returns Klein 4B (bonsai-image-binary-4B) defaults.
func DefaultConfig() Config {
	return Config{
		PatchSize:                1,
		InChannels:               128,
		OutChannels:              128,
		NumLayers:                5,
		NumSingleLayers:          20,
		AttentionHeadDim:         128,
		NumAttentionHeads:        24,
		JointAttentionDim:        7680,
		TimestepGuidanceChannels: 256,
		MLPRatio:                 3.0,
		AxesDimsRope:             []int{32, 32, 32, 32},
		RopeTheta:                2000,
		Eps:                      1e-6,
		GuidanceEmbeds:           false,
	}
}

// LoadConfig reads transformer-packed-mflux/config.json (or a direct config.json path).
func LoadConfig(snapshotDir string) (Config, error) {
	cfg := DefaultConfig()
	candidates := []string{
		filepath.Join(snapshotDir, "transformer-packed-mflux", "config.json"),
		filepath.Join(snapshotDir, "transformer", "config.json"),
		filepath.Join(snapshotDir, "config.json"),
		snapshotDir, // allow passing config.json path directly
	}
	var data []byte
	var err error
	var used string
	for _, p := range candidates {
		data, err = os.ReadFile(p)
		if err == nil {
			used = p
			break
		}
	}
	if data == nil {
		return cfg, fmt.Errorf("flux2.LoadConfig: no config.json under %s", snapshotDir)
	}
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		return cfg, fmt.Errorf("flux2.LoadConfig %s: %w", used, err)
	}
	applyConfigJSON(&cfg, raw)
	if cfg.OutChannels == 0 {
		cfg.OutChannels = cfg.InChannels
	}
	return cfg, nil
}

func applyConfigJSON(cfg *Config, raw map[string]any) {
	if v, ok := asInt(raw["patch_size"]); ok {
		cfg.PatchSize = v
	}
	if v, ok := asInt(raw["in_channels"]); ok {
		cfg.InChannels = v
	}
	if v, ok := asInt(raw["out_channels"]); ok {
		cfg.OutChannels = v
	}
	if v, ok := asInt(raw["num_layers"]); ok {
		cfg.NumLayers = v
	}
	if v, ok := asInt(raw["num_single_layers"]); ok {
		cfg.NumSingleLayers = v
	}
	if v, ok := asInt(raw["attention_head_dim"]); ok {
		cfg.AttentionHeadDim = v
	}
	if v, ok := asInt(raw["num_attention_heads"]); ok {
		cfg.NumAttentionHeads = v
	}
	if v, ok := asInt(raw["joint_attention_dim"]); ok {
		cfg.JointAttentionDim = v
	}
	if v, ok := asInt(raw["timestep_guidance_channels"]); ok {
		cfg.TimestepGuidanceChannels = v
	}
	if v, ok := asFloat(raw["mlp_ratio"]); ok {
		cfg.MLPRatio = v
	}
	if v, ok := asFloat(raw["rope_theta"]); ok {
		cfg.RopeTheta = v
	}
	if v, ok := asFloat(raw["eps"]); ok {
		cfg.Eps = v
	}
	if v, ok := raw["guidance_embeds"].(bool); ok {
		cfg.GuidanceEmbeds = v
	}
	if arr, ok := raw["axes_dims_rope"].([]any); ok && len(arr) > 0 {
		dims := make([]int, 0, len(arr))
		for _, x := range arr {
			if n, ok := asInt(x); ok {
				dims = append(dims, n)
			}
		}
		if len(dims) > 0 {
			cfg.AxesDimsRope = dims
		}
	}
}

func asInt(v any) (int, bool) {
	switch t := v.(type) {
	case float64:
		return int(t), true
	case int:
		return t, true
	case int64:
		return int(t), true
	case json.Number:
		i, err := t.Int64()
		return int(i), err == nil
	default:
		return 0, false
	}
}

func asFloat(v any) (float64, bool) {
	switch t := v.(type) {
	case float64:
		return t, true
	case int:
		return float64(t), true
	case json.Number:
		f, err := t.Float64()
		return f, err == nil
	default:
		return 0, false
	}
}
