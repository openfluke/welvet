package hf

import (
	"fmt"
	"strings"
)

// DecoderDims holds Llama-style decoder hyperparameters from config.json.
type DecoderDims struct {
	NumLayers        int
	HiddenSize       int
	NumHeads         int
	NumKVHeads       int
	HeadDim          int
	QueryDim         int
	KVDim            int
	IntermediateSize int
	VocabSize        int
	RMSNormEps       float64
	RoPEFreqBase     float64
}

// ParseDecoderDims extracts decoder dimensions from config (+ optional safetensors for layer count).
func ParseDecoderDims(config map[string]any, safetensorFiles []string) (DecoderDims, error) {
	config = EffectiveConfig(config)
	numHeads, ok := ConfigInt(config, "num_attention_heads")
	if !ok || numHeads <= 0 {
		return DecoderDims{}, fmt.Errorf("config missing num_attention_heads")
	}
	numKVHeads := numHeads
	if v, ok := ConfigInt(config, "num_key_value_heads"); ok && v > 0 {
		numKVHeads = v
	}
	hiddenSize, ok := ConfigInt(config, "hidden_size")
	if !ok || hiddenSize <= 0 {
		return DecoderDims{}, fmt.Errorf("config missing hidden_size")
	}
	intermediateSize, ok := ConfigInt(config, "intermediate_size")
	if !ok || intermediateSize <= 0 {
		return DecoderDims{}, fmt.Errorf("config missing intermediate_size")
	}
	numLayers, ok := ConfigInt(config, "num_hidden_layers")
	if !ok || numLayers <= 0 {
		maxLi := MaxWeightLayerIndexInFiles(safetensorFiles)
		if maxLi < 0 {
			return DecoderDims{}, fmt.Errorf("could not determine num_hidden_layers")
		}
		numLayers = maxLi + 1
	}
	headDim := hiddenSize / numHeads
	if v, ok := ConfigInt(config, "head_dim"); ok && v > 0 {
		headDim = v
	}
	vocab := ConfigIntDefault(config, "vocab_size", 0)
	return DecoderDims{
		NumLayers:        numLayers,
		HiddenSize:       hiddenSize,
		NumHeads:         numHeads,
		NumKVHeads:       numKVHeads,
		HeadDim:          headDim,
		QueryDim:         numHeads * headDim,
		KVDim:            numKVHeads * headDim,
		IntermediateSize: intermediateSize,
		VocabSize:        vocab,
		RMSNormEps:       ConfigFloat64Default(config, "rms_norm_eps", 1e-6),
		RoPEFreqBase:     ConfigFloat64Default(config, "rope_theta", 10000.0),
	}, nil
}

// WeightLayerIndex returns the transformer block index for HF keys, or ok=false for globals.
func WeightLayerIndex(key string) (idx int, ok bool) {
	parts := strings.Split(key, ".")
	for i := 0; i < len(parts)-1; i++ {
		if parts[i] == "layers" || parts[i] == "h" {
			var n int
			if _, err := fmt.Sscanf(parts[i+1], "%d", &n); err == nil {
				return n, true
			}
		}
	}
	return 0, false
}

// WeightIsGlobal reports whether key is not tied to a numbered block.
func WeightIsGlobal(key string) bool {
	_, ok := WeightLayerIndex(key)
	return !ok
}

// WeightMatchesLayer reports whether key belongs to block layerIdx.
func WeightMatchesLayer(key string, layerIdx int) bool {
	n, ok := WeightLayerIndex(key)
	return ok && n == layerIdx
}

// MaxWeightLayerIndex returns the largest block index in names, or -1.
func MaxWeightLayerIndex(names []string) int {
	max := -1
	for _, n := range names {
		if li, ok := WeightLayerIndex(n); ok && li > max {
			max = li
		}
	}
	return max
}

// MaxWeightLayerIndexInFiles scans safetensors headers for the max layer index.
func MaxWeightLayerIndexInFiles(paths []string) int {
	max := -1
	for _, p := range paths {
		names, err := TensorNames(p)
		if err != nil {
			continue
		}
		if m := MaxWeightLayerIndex(names); m > max {
			max = m
		}
	}
	return max
}

// BuildLayerShardIndex maps each block index to safetensors files containing its tensors.
func BuildLayerShardIndex(safetensorFiles []string, numLayers int) [][]string {
	layerFiles := make([][]string, numLayers)
	if numLayers <= 0 {
		return layerFiles
	}
	for _, sf := range safetensorFiles {
		names, err := TensorNames(sf)
		if err != nil {
			for li := 0; li < numLayers; li++ {
				layerFiles[li] = append(layerFiles[li], sf)
			}
			continue
		}
		seen := make(map[int]struct{})
		for _, n := range names {
			if li, ok := WeightLayerIndex(n); ok && li >= 0 && li < numLayers {
				seen[li] = struct{}{}
			}
		}
		for li := range seen {
			layerFiles[li] = append(layerFiles[li], sf)
		}
	}
	for li := 0; li < numLayers; li++ {
		if len(layerFiles[li]) == 0 {
			layerFiles[li] = append(layerFiles[li], safetensorFiles...)
		}
	}
	return layerFiles
}
