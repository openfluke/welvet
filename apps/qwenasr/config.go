// Package qwenasr runs Qwen3-ASR checkpoints directly from Hugging Face files.
package qwenasr

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

const (
	audioStartID = 151669
	audioEndID   = 151670
	audioTokenID = 151676
	asrTextID    = 151704
	imStartID    = 151644
	imEndID      = 151645
	eosID        = 151643
)

var promptPrefix = []int{imStartID, 8948, 198, imEndID, 198, imStartID, 872, 198, audioStartID}
var promptSuffix = []int{audioEndID, imEndID, 198, imStartID, 77091, 198}

type EncoderConfig struct {
	DModel, Layers, Heads, FFN int
	DownsampleHidden, NumMel   int
	NWindow, NWindowInfer      int
	OutputDim, ConvChunkSize   int
}
type DecoderConfig struct {
	Hidden, Layers, Heads, KVHeads, HeadDim, Intermediate, Vocab int
	RMSEps, RopeTheta                                            float64
}
type Config struct {
	Encoder EncoderConfig
	Decoder DecoderConfig
}

type rawConfig struct {
	Thinker struct {
		Audio struct {
			DModel json.Number `json:"d_model"`
			Layers json.Number `json:"encoder_layers"`
			Heads  json.Number `json:"encoder_attention_heads"`
			FFN    json.Number `json:"encoder_ffn_dim"`
			Down   json.Number `json:"downsample_hidden_size"`
			Mel    json.Number `json:"num_mel_bins"`
			Window json.Number `json:"n_window"`
			Infer  json.Number `json:"n_window_infer"`
			Output json.Number `json:"output_dim"`
			Chunk  json.Number `json:"conv_chunksize"`
		} `json:"audio_config"`
		Text struct {
			Hidden       json.Number `json:"hidden_size"`
			Layers       json.Number `json:"num_hidden_layers"`
			Heads        json.Number `json:"num_attention_heads"`
			KVHeads      json.Number `json:"num_key_value_heads"`
			HeadDim      json.Number `json:"head_dim"`
			Intermediate json.Number `json:"intermediate_size"`
			Eps          json.Number `json:"rms_norm_eps"`
			Theta        json.Number `json:"rope_theta"`
			Vocab        json.Number `json:"vocab_size"`
		} `json:"text_config"`
	} `json:"thinker_config"`
}

func cfgInt(n json.Number, d int) int {
	if n == "" {
		return d
	}
	v, e := n.Int64()
	if e != nil {
		return d
	}
	return int(v)
}
func cfgFloat(n json.Number, d float64) float64 {
	if n == "" {
		return d
	}
	v, e := n.Float64()
	if e != nil {
		return d
	}
	return v
}

func LoadConfig(snap string) (*Config, error) {
	b, err := os.ReadFile(filepath.Join(snap, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("qwenasr config: %w", err)
	}
	var raw rawConfig
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	if err := dec.Decode(&raw); err != nil {
		return nil, fmt.Errorf("qwenasr config parse: %w", err)
	}
	a, t := raw.Thinker.Audio, raw.Thinker.Text
	c := &Config{
		Encoder: EncoderConfig{cfgInt(a.DModel, 896), cfgInt(a.Layers, 18), cfgInt(a.Heads, 14), cfgInt(a.FFN, 3584), cfgInt(a.Down, 480), cfgInt(a.Mel, 128), cfgInt(a.Window, 50), cfgInt(a.Infer, 800), cfgInt(a.Output, 1024), cfgInt(a.Chunk, 500)},
		Decoder: DecoderConfig{cfgInt(t.Hidden, 1024), cfgInt(t.Layers, 28), cfgInt(t.Heads, 16), cfgInt(t.KVHeads, 8), cfgInt(t.HeadDim, 128), cfgInt(t.Intermediate, 3072), cfgInt(t.Vocab, 151936), cfgFloat(t.Eps, 1e-6), cfgFloat(t.Theta, 1e6)},
	}
	if c.Encoder.DModel%c.Encoder.Heads != 0 || c.Decoder.Heads%c.Decoder.KVHeads != 0 {
		return nil, fmt.Errorf("qwenasr config: incompatible attention dimensions")
	}
	return c, nil
}
