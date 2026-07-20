package hf

import (
	"fmt"
	"os"
	"path/filepath"
)

// SnapshotInfo is the probe result for a Hugging Face checkpoint directory
// (config + dims + safetensors inventory) before ENTITY packing.
type SnapshotInfo struct {
	Architecture    ArchitectureKind
	Config          map[string]any
	Dims            DecoderDims
	EOSTokens       []int
	SafetensorFiles []string
	Hybrid          bool // Qwen3.5 / Bonsai GDN+attn tower
}

// InspectSnapshot probes an HF snapshot directory without loading weight tensors.
// Use entity.PackFromHF / ImportFromHF to bake a Welvet .entity afterward.
func InspectSnapshot(snapshotDir string) (*SnapshotInfo, error) {
	configPath := filepath.Join(snapshotDir, "config.json")
	if _, err := os.Stat(configPath); err != nil {
		return nil, fmt.Errorf("hf: config.json: %w", err)
	}
	config, err := LoadConfigJSON(configPath)
	if err != nil {
		return nil, fmt.Errorf("hf: config.json: %w", err)
	}
	kind := DetectArchitecture(config)
	if kind == ArchUnknown {
		return nil, fmt.Errorf("hf: unsupported architecture (model_type=%q)",
			ConfigStringDefault(config, "model_type", ""))
	}
	sts, err := filepath.Glob(filepath.Join(snapshotDir, "*.safetensors"))
	if err != nil {
		return nil, err
	}
	dims, err := ParseDecoderDims(config, sts)
	if err != nil {
		return nil, fmt.Errorf("hf: dims: %w", err)
	}
	return &SnapshotInfo{
		Architecture:    kind,
		Config:          config,
		Dims:            dims,
		EOSTokens:       EOSTokenIDs(config),
		SafetensorFiles: sts,
		Hybrid:          kind == ArchQwen35Hybrid,
	}, nil
}
