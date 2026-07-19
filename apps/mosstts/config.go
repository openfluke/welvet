// Package mosstts is a native Welvet port of MOSS-TTS-Nano (global AR + local RVQ + audio codec).
//
// Tests live in w2a — not here.
package mosstts

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Config matches MossTTSNanoConfig / nested gpt2_config (inference fields only).
type Config struct {
	HiddenSize              int     `json:"hidden_size"`
	VocabSize               int     `json:"vocab_size"`
	NVQ                     int     `json:"n_vq"`
	AudioVocabSize          int     `json:"audio_vocab_size"`
	AudioPadTokenID         int     `json:"audio_pad_token_id"`
	AudioStartTokenID       int     `json:"audio_start_token_id"`
	AudioEndTokenID         int     `json:"audio_end_token_id"`
	AudioAssistantSlotID    int     `json:"audio_assistant_slot_token_id"`
	ImStartTokenID          int     `json:"im_start_token_id"`
	ImEndTokenID            int     `json:"im_end_token_id"`
	PadTokenID              int     `json:"pad_token_id"`
	LocalTransformerLayers  int     `json:"local_transformer_layers"`
	AudioTokenizerSampleRate int    `json:"audio_tokenizer_sample_rate"`
	GPT2                    GPT2Cfg `json:"gpt2_config"`
	AudioCodebookSizes      []int   `json:"audio_codebook_sizes"`
}

// GPT2Cfg is the global (and local) GPT2-like backbone config.
type GPT2Cfg struct {
	NLayer               int     `json:"n_layer"`
	NEmbd                int     `json:"n_embd"`
	NHead                int     `json:"n_head"`
	NInner               int     `json:"n_inner"`
	NPositions           int     `json:"n_positions"`
	VocabSize            int     `json:"vocab_size"`
	LayerNormEps         float64 `json:"layer_norm_epsilon"`
	ActivationFunction   string  `json:"activation_function"`
	PositionEmbeddingType string `json:"position_embedding_type"`
	RopeBase             float64 `json:"rope_base"`
	ScaleAttnWeights     bool    `json:"scale_attn_weights"`
	PadTokenID           int     `json:"pad_token_id"`
}

// LoadConfig reads snapshot config.json.
func LoadConfig(snapshotDir string) (*Config, error) {
	data, err := os.ReadFile(filepath.Join(snapshotDir, "config.json"))
	if err != nil {
		return nil, err
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("mosstts config: %w", err)
	}
	if cfg.HiddenSize == 0 {
		cfg.HiddenSize = cfg.GPT2.NEmbd
	}
	if cfg.VocabSize == 0 {
		cfg.VocabSize = cfg.GPT2.VocabSize
	}
	if cfg.NVQ == 0 {
		cfg.NVQ = 16
	}
	if cfg.LocalTransformerLayers == 0 {
		cfg.LocalTransformerLayers = 1
	}
	if cfg.GPT2.NInner == 0 {
		cfg.GPT2.NInner = 4 * cfg.GPT2.NEmbd
	}
	if cfg.GPT2.RopeBase == 0 {
		cfg.GPT2.RopeBase = 10000
	}
	if cfg.GPT2.LayerNormEps == 0 {
		cfg.GPT2.LayerNormEps = 1e-5
	}
	cfg.GPT2.ScaleAttnWeights = true
	if cfg.AudioTokenizerSampleRate == 0 {
		cfg.AudioTokenizerSampleRate = 48000
	}
	if len(cfg.AudioCodebookSizes) == 0 {
		cfg.AudioCodebookSizes = make([]int, cfg.NVQ)
		for i := range cfg.AudioCodebookSizes {
			cfg.AudioCodebookSizes[i] = cfg.AudioVocabSize
			if cfg.AudioCodebookSizes[i] == 0 {
				cfg.AudioCodebookSizes[i] = 1024
			}
		}
	}
	return &cfg, nil
}
