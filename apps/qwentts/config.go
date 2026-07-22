package qwentts

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Well-known special token IDs for Qwen3-TTS-12Hz (CustomVoice/VoiceDesign).
// These are fixed for the model family; config.json may override where present.
const (
	tokIMStart   = 151644
	tokAssistant = 77091
	tokNewline   = 198
	tokIMEnd     = 151645
	tokEndOfText = 151643

	ttsBOS = 151672
	ttsEOS = 151673
	ttsPAD = 151671

	codecPAD      = 2148
	codecBOS      = 2149
	codecEOS      = 2150
	codecThink    = 2154
	codecNoThink  = 2155
	codecThinkBOS = 2156
	codecThinkEOS = 2157
)

// defaultLangID maps a lowercased language name to its codec vocab id.
var defaultLangID = map[string]int{
	"chinese": 2055, "english": 2050, "japanese": 2058, "korean": 2064,
	"german": 2053, "french": 2061, "russian": 2069, "portuguese": 2071,
	"spanish": 2054, "italian": 2070,
}

// defaultSpkID maps a lowercased speaker name to its codec vocab id.
var defaultSpkID = map[string]int{
	"serena": 3066, "vivian": 3065, "uncle_fu": 3010, "ryan": 3061,
	"aiden": 2861, "ono_anna": 2873, "sohee": 2864, "eric": 2875, "dylan": 2878,
}

// TalkerConfig holds the Qwen3 talker backbone dims.
type TalkerConfig struct {
	HiddenSize       int
	TextHiddenSize   int
	NumLayers        int
	NumHeads         int
	NumKVHeads       int
	HeadDim          int
	IntermediateSize int
	CodecVocabSize   int
	CodebookSize     int
	TextVocabSize    int
	NumCodeGroups    int
	RMSNormEps       float64
	RopeTheta        float64
	MaxPos           int
}

// CodePredictorConfig holds the MTP (code predictor) dims.
type CodePredictorConfig struct {
	HiddenSize       int
	NumLayers        int
	NumHeads         int
	NumKVHeads       int
	HeadDim          int
	IntermediateSize int
	VocabSize        int
	RMSNormEps       float64
	RopeTheta        float64
	MaxPos           int
}

// DecoderConfig holds the speech-tokenizer decoder (vocoder) dims.
type DecoderConfig struct {
	HiddenSize       int
	NumLayers        int
	LatentDim        int
	CodebookDim      int
	DecoderDim       int
	NumHeads         int
	HeadDim          int
	IntermediateSize int
	NumQuantizers    int
	SlidingWindow    int
	RopeTheta        float64
	RMSNormEps       float64
	CodebookSize     int
	UpsampleRates    []int // Snake decoder upsample rates, e.g. [8,5,4,3]
	UpsamplingRatios []int // ConvNeXt upsample ratios, e.g. [2,2]
	SampleRate       int
}

// Config is the parsed top-level Qwen3-TTS config plus nested sub-configs.
type Config struct {
	TTSModelType  string
	TTSModelSize  string
	TokenizerType string
	TTSBosID      int
	TTSEosID      int
	TTSPadID      int

	Talker        TalkerConfig
	CodePredictor CodePredictorConfig
	Decoder       DecoderConfig

	SpkID        map[string]int
	LangID       map[string]int
	SpkIsDialect map[string]string

	// Codec special-token ids (fixed defaults, overridable).
	CodecPAD, CodecBOS, CodecEOS int
	CodecThink, CodecNoThink     int
	CodecThinkBOS, CodecThinkEOS int
}

// rawTalker mirrors the fields we read out of talker_config.
type rawTalker struct {
	HiddenSize          json.Number `json:"hidden_size"`
	TextHiddenSize      json.Number `json:"text_hidden_size"`
	NumHiddenLayers     json.Number `json:"num_hidden_layers"`
	NumAttentionHeads   json.Number `json:"num_attention_heads"`
	NumKeyValueHeads    json.Number `json:"num_key_value_heads"`
	HeadDim             json.Number `json:"head_dim"`
	IntermediateSize    json.Number `json:"intermediate_size"`
	CodecVocabSize      json.Number `json:"codec_vocab_size"`
	VocabSize           json.Number `json:"vocab_size"`
	CodebookSize        json.Number `json:"codebook_size"`
	TextVocabSize       json.Number `json:"text_vocab_size"`
	NumCodeGroups       json.Number `json:"num_code_groups"`
	RMSNormEps          json.Number `json:"rms_norm_eps"`
	RopeTheta           json.Number `json:"rope_theta"`
	MaxPositionEmbeds   json.Number `json:"max_position_embeddings"`
	SpkID               map[string]int             `json:"spk_id"`
	CodecLanguageID     map[string]int             `json:"codec_language_id"`
	SpkIsDialect        map[string]json.RawMessage `json:"spk_is_dialect"`
	CodePredictorConfig json.RawMessage            `json:"code_predictor_config"`
	// codec special tokens (some snapshots include them)
	CodecPadID      json.Number `json:"codec_pad_id"`
	CodecBosID      json.Number `json:"codec_bos_id"`
	CodecEosTokenID json.Number `json:"codec_eos_token_id"`
	CodecThinkID    json.Number `json:"codec_think_id"`
	CodecNoThinkID  json.Number `json:"codec_nothink_id"`
	CodecThinkBosID json.Number `json:"codec_think_bos_id"`
	CodecThinkEosID json.Number `json:"codec_think_eos_id"`
}

type rawCP struct {
	HiddenSize        json.Number `json:"hidden_size"`
	NumHiddenLayers   json.Number `json:"num_hidden_layers"`
	NumAttentionHeads json.Number `json:"num_attention_heads"`
	NumKeyValueHeads  json.Number `json:"num_key_value_heads"`
	HeadDim           json.Number `json:"head_dim"`
	IntermediateSize  json.Number `json:"intermediate_size"`
	VocabSize         json.Number `json:"vocab_size"`
	RMSNormEps        json.Number `json:"rms_norm_eps"`
	RopeTheta         json.Number `json:"rope_theta"`
	MaxPositionEmbeds json.Number `json:"max_position_embeddings"`
}

type rawTop struct {
	TTSModelType  string          `json:"tts_model_type"`
	TTSModelSize  string          `json:"tts_model_size"`
	TokenizerType string          `json:"tokenizer_type"`
	TTSBosTokenID json.Number     `json:"tts_bos_token_id"`
	TTSEosTokenID json.Number     `json:"tts_eos_token_id"`
	TTSPadTokenID json.Number     `json:"tts_pad_token_id"`
	TalkerConfig  json.RawMessage `json:"talker_config"`
}

type rawDecoder struct {
	HiddenSize       json.Number `json:"hidden_size"`
	NumHiddenLayers  json.Number `json:"num_hidden_layers"`
	LatentDim        json.Number `json:"latent_dim"`
	CodebookDim      json.Number `json:"codebook_dim"`
	DecoderDim       json.Number `json:"decoder_dim"`
	NumHeads         json.Number `json:"num_attention_heads"`
	HeadDim          json.Number `json:"head_dim"`
	IntermediateSize json.Number `json:"intermediate_size"`
	NumQuantizers    json.Number `json:"num_quantizers"`
	SlidingWindow    json.Number `json:"sliding_window"`
	RopeTheta        json.Number `json:"rope_theta"`
	RMSNormEps       json.Number `json:"rms_norm_eps"`
	CodebookSize     json.Number `json:"codebook_size"`
	UpsampleRates    []int       `json:"upsample_rates"`
	UpsamplingRatios []int       `json:"upsampling_ratios"`
	SampleRate       json.Number `json:"output_sample_rate"`
}

func numInt(n json.Number, def int) int {
	if n == "" {
		return def
	}
	i, err := n.Int64()
	if err != nil {
		f, ferr := n.Float64()
		if ferr != nil {
			return def
		}
		return int(f)
	}
	return int(i)
}

func numFloat(n json.Number, def float64) float64 {
	if n == "" {
		return def
	}
	f, err := n.Float64()
	if err != nil {
		return def
	}
	return f
}

// LoadConfig parses config.json (+ nested talker/code_predictor) and
// speech_tokenizer/config.json from the snapshot directory.
func LoadConfig(snapshotDir string) (*Config, error) {
	data, err := os.ReadFile(filepath.Join(snapshotDir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("qwentts config: %w", err)
	}
	var top rawTop
	if err := json.Unmarshal(data, &top); err != nil {
		return nil, fmt.Errorf("qwentts config parse: %w", err)
	}
	if len(top.TalkerConfig) == 0 {
		return nil, fmt.Errorf("qwentts config: missing talker_config")
	}
	var tk rawTalker
	if err := json.Unmarshal(top.TalkerConfig, &tk); err != nil {
		return nil, fmt.Errorf("qwentts talker_config parse: %w", err)
	}

	cfg := &Config{
		TTSModelType:  strings.ToLower(top.TTSModelType),
		TTSModelSize:  top.TTSModelSize,
		TTSBosID:      numInt(top.TTSBosTokenID, ttsBOS),
		TTSEosID:      numInt(top.TTSEosTokenID, ttsEOS),
		TTSPadID:      numInt(top.TTSPadTokenID, ttsPAD),
		CodecPAD:      numInt(tk.CodecPadID, codecPAD),
		CodecBOS:      numInt(tk.CodecBosID, codecBOS),
		CodecEOS:      numInt(tk.CodecEosTokenID, codecEOS),
		CodecThink:    numInt(tk.CodecThinkID, codecThink),
		CodecNoThink:  numInt(tk.CodecNoThinkID, codecNoThink),
		CodecThinkBOS: numInt(tk.CodecThinkBosID, codecThinkBOS),
		CodecThinkEOS: numInt(tk.CodecThinkEosID, codecThinkEOS),
	}
	cfg.TokenizerType = top.TokenizerType

	cfg.Talker = TalkerConfig{
		HiddenSize:       numInt(tk.HiddenSize, 1024),
		TextHiddenSize:   numInt(tk.TextHiddenSize, 2048),
		NumLayers:        numInt(tk.NumHiddenLayers, 28),
		NumHeads:         numInt(tk.NumAttentionHeads, 16),
		NumKVHeads:       numInt(tk.NumKeyValueHeads, 8),
		HeadDim:          numInt(tk.HeadDim, 128),
		IntermediateSize: numInt(tk.IntermediateSize, 3072),
		CodecVocabSize:   numInt(tk.CodecVocabSize, 3072),
		CodebookSize:     numInt(tk.CodebookSize, 2048),
		TextVocabSize:    numInt(tk.TextVocabSize, numInt(tk.VocabSize, 151936)),
		NumCodeGroups:    numInt(tk.NumCodeGroups, 16),
		RMSNormEps:       numFloat(tk.RMSNormEps, 1e-6),
		RopeTheta:        numFloat(tk.RopeTheta, 1e6),
		MaxPos:           numInt(tk.MaxPositionEmbeds, 8192),
	}

	// code_predictor_config (nested inside talker_config)
	cp := rawCP{}
	if len(tk.CodePredictorConfig) > 0 {
		_ = json.Unmarshal(tk.CodePredictorConfig, &cp)
	}
	cfg.CodePredictor = CodePredictorConfig{
		HiddenSize:       numInt(cp.HiddenSize, cfg.Talker.HiddenSize),
		NumLayers:        numInt(cp.NumHiddenLayers, 5),
		NumHeads:         numInt(cp.NumAttentionHeads, cfg.Talker.NumHeads),
		NumKVHeads:       numInt(cp.NumKeyValueHeads, cfg.Talker.NumKVHeads),
		HeadDim:          numInt(cp.HeadDim, cfg.Talker.HeadDim),
		IntermediateSize: numInt(cp.IntermediateSize, cfg.Talker.IntermediateSize),
		VocabSize:        numInt(cp.VocabSize, cfg.Talker.CodebookSize),
		RMSNormEps:       numFloat(cp.RMSNormEps, cfg.Talker.RMSNormEps),
		RopeTheta:        numFloat(cp.RopeTheta, cfg.Talker.RopeTheta),
		MaxPos:           numInt(cp.MaxPositionEmbeds, 64),
	}

	// Speaker / language maps.
	cfg.SpkID = map[string]int{}
	for k, v := range defaultSpkID {
		cfg.SpkID[k] = v
	}
	for k, v := range tk.SpkID {
		cfg.SpkID[strings.ToLower(k)] = v
	}
	cfg.LangID = map[string]int{}
	for k, v := range defaultLangID {
		cfg.LangID[k] = v
	}
	for k, v := range tk.CodecLanguageID {
		cfg.LangID[strings.ToLower(k)] = v
	}
	cfg.SpkIsDialect = map[string]string{}
	for k, v := range tk.SpkIsDialect {
		s := strings.Trim(string(v), `"`)
		if s == "false" || s == "" || s == "null" {
			continue
		}
		cfg.SpkIsDialect[strings.ToLower(k)] = s
	}

	// speech_tokenizer/config.json -> decoder_config
	cfg.Decoder = DecoderConfig{
		HiddenSize:       512,
		NumLayers:        8,
		LatentDim:        1024,
		CodebookDim:      512,
		DecoderDim:       1536,
		NumHeads:         16,
		HeadDim:          64,
		IntermediateSize: 1024,
		NumQuantizers:    16,
		SlidingWindow:    72,
		RopeTheta:        10000,
		RMSNormEps:       1e-5,
		CodebookSize:     2048,
		UpsampleRates:    []int{8, 5, 4, 3},
		UpsamplingRatios: []int{2, 2},
		SampleRate:       24000,
	}
	if stData, err := os.ReadFile(filepath.Join(snapshotDir, "speech_tokenizer", "config.json")); err == nil {
		var stTop struct {
			DecoderConfig    json.RawMessage `json:"decoder_config"`
			OutputSampleRate json.Number     `json:"output_sample_rate"`
		}
		if json.Unmarshal(stData, &stTop) == nil && len(stTop.DecoderConfig) > 0 {
			var dc rawDecoder
			if json.Unmarshal(stTop.DecoderConfig, &dc) == nil {
				applyDecoder(&cfg.Decoder, dc)
			}
			if sr := numInt(stTop.OutputSampleRate, 0); sr > 0 {
				cfg.Decoder.SampleRate = sr
			}
		}
	}
	return cfg, nil
}

func applyDecoder(d *DecoderConfig, dc rawDecoder) {
	d.HiddenSize = numInt(dc.HiddenSize, d.HiddenSize)
	d.NumLayers = numInt(dc.NumHiddenLayers, d.NumLayers)
	d.LatentDim = numInt(dc.LatentDim, d.LatentDim)
	d.CodebookDim = numInt(dc.CodebookDim, d.CodebookDim)
	d.DecoderDim = numInt(dc.DecoderDim, d.DecoderDim)
	d.NumHeads = numInt(dc.NumHeads, d.NumHeads)
	d.HeadDim = numInt(dc.HeadDim, d.HeadDim)
	d.IntermediateSize = numInt(dc.IntermediateSize, d.IntermediateSize)
	d.NumQuantizers = numInt(dc.NumQuantizers, d.NumQuantizers)
	d.SlidingWindow = numInt(dc.SlidingWindow, d.SlidingWindow)
	d.RopeTheta = numFloat(dc.RopeTheta, d.RopeTheta)
	d.RMSNormEps = numFloat(dc.RMSNormEps, d.RMSNormEps)
	d.CodebookSize = numInt(dc.CodebookSize, d.CodebookSize)
	if len(dc.UpsampleRates) > 0 {
		d.UpsampleRates = dc.UpsampleRates
	}
	if len(dc.UpsamplingRatios) > 0 {
		d.UpsamplingRatios = dc.UpsamplingRatios
	}
	if sr := numInt(dc.SampleRate, 0); sr > 0 {
		d.SampleRate = sr
	}
}
