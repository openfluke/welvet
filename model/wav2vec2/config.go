package wav2vec2

import (
	"encoding/json"
	"fmt"
	"os"
)

// Config mirrors the HF wav2vec2-base-960h fields we need for inference.
type Config struct {
	HiddenSize               int   `json:"hidden_size"`
	NumHiddenLayers          int   `json:"num_hidden_layers"`
	NumAttentionHeads        int   `json:"num_attention_heads"`
	IntermediateSize         int   `json:"intermediate_size"`
	ConvDim                  []int `json:"conv_dim"`
	ConvKernel               []int `json:"conv_kernel"`
	ConvStride               []int `json:"conv_stride"`
	ConvBias                 bool  `json:"conv_bias"`
	NumConvPosEmbeddings     int   `json:"num_conv_pos_embeddings"`
	NumConvPosEmbeddingGroups int  `json:"num_conv_pos_embedding_groups"`
	LayerNormEps             float64 `json:"layer_norm_eps"`
	VocabSize                int   `json:"vocab_size"`
	PadTokenID               int   `json:"pad_token_id"`
	DoStableLayerNorm        bool  `json:"do_stable_layer_norm"`
	FeatExtractNorm          string `json:"feat_extract_norm"`
}

// Base960h returns the facebook/wav2vec2-base-960h defaults.
func Base960h() Config {
	return Config{
		HiddenSize:                768,
		NumHiddenLayers:           12,
		NumAttentionHeads:         12,
		IntermediateSize:          3072,
		ConvDim:                   []int{512, 512, 512, 512, 512, 512, 512},
		ConvKernel:                []int{10, 3, 3, 3, 3, 2, 2},
		ConvStride:                []int{5, 2, 2, 2, 2, 2, 2},
		ConvBias:                  false,
		NumConvPosEmbeddings:      128,
		NumConvPosEmbeddingGroups: 16,
		LayerNormEps:              1e-5,
		VocabSize:                 32,
		PadTokenID:                0,
		DoStableLayerNorm:         false,
		FeatExtractNorm:           "group",
	}
}

// LoadConfigJSON reads a HF config.json.
func LoadConfigJSON(path string) (Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Config{}, err
	}
	cfg := Base960h()
	if err := json.Unmarshal(b, &cfg); err != nil {
		return Config{}, fmt.Errorf("wav2vec2: config: %w", err)
	}
	if cfg.HiddenSize == 0 || cfg.NumHiddenLayers == 0 {
		return Config{}, fmt.Errorf("wav2vec2: incomplete config")
	}
	return cfg, nil
}

func (c Config) headDim() int {
	return c.HiddenSize / c.NumAttentionHeads
}

func (c Config) featOutDim() int {
	if len(c.ConvDim) == 0 {
		return 0
	}
	return c.ConvDim[len(c.ConvDim)-1]
}
