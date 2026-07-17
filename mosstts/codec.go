package mosstts

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/openfluke/welvet/hf"
)

// AudioTokenizerConfig is the HF MossAudioTokenizerConfig (decode fields).
type AudioTokenizerConfig struct {
	SampleRate             int                      `json:"sample_rate"`
	SamplingRate           int                      `json:"sampling_rate"`
	DownsampleRate         int                      `json:"downsample_rate"`
	NumberChannels         int                      `json:"number_channels"`
	EnableChannelInterleave bool                    `json:"enable_channel_interleave"`
	CausalCtxDuration      float64                  `json:"causal_transformer_context_duration"`
	DecoderKwargs          []map[string]any         `json:"decoder_kwargs"`
	QuantizerKwargs        map[string]any           `json:"quantizer_kwargs"`
}

func (c *AudioTokenizerConfig) SR() int {
	if c.SamplingRate > 0 {
		return c.SamplingRate
	}
	if c.SampleRate > 0 {
		return c.SampleRate
	}
	return 48000
}

// AudioTokenizer is the decode-only MOSS-Audio-Tokenizer-Nano.
type AudioTokenizer struct {
	Cfg        *AudioTokenizerConfig
	Quant      *residualLFQ
	Decoder    []decoderModule
	Channels   int
	Interleave bool
}

type decoderModule interface {
	Forward(x []float32, channels, length int) (out []float32, outCh, outLen int)
}

// LoadAudioTokenizer loads decode weights from a tokenizer snapshot directory.
func LoadAudioTokenizer(dir string) (*AudioTokenizer, error) {
	cfgPath := filepath.Join(dir, "config.json")
	data, err := os.ReadFile(cfgPath)
	if err != nil {
		return nil, err
	}
	var cfg AudioTokenizerConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("audio tokenizer config: %w", err)
	}
	if cfg.NumberChannels == 0 {
		cfg.NumberChannels = 1
	}
	stPath, err := findSTFile(dir)
	if err != nil {
		return nil, err
	}
	tensors, err := hf.LoadSafetensorsSelective(stPath, func(name string) bool {
		return strings.HasPrefix(name, "decoder.") || strings.HasPrefix(name, "quantizer.")
	})
	if err != nil {
		return nil, err
	}

	qk := cfg.QuantizerKwargs
	inputDim := intFromAny(qk["input_dim"], 768)
	rvqDim := intFromAny(qk["rvq_dim"], 512)
	outputDim := intFromAny(qk["output_dim"], 768)
	nQ := intFromAny(qk["num_quantizers"], 16)
	cbSize := intFromAny(qk["codebook_size"], 1024)
	cbDim := intFromAny(qk["codebook_dim"], 8)

	quant, err := loadResidualLFQ(tensors, "quantizer", inputDim, rvqDim, outputDim, nQ, cbSize, cbDim)
	if err != nil {
		return nil, err
	}

	interleaveFactor := 1
	if cfg.EnableChannelInterleave && cfg.NumberChannels > 1 {
		interleaveFactor = cfg.NumberChannels
	}
	frameRate := float64(cfg.SR() * interleaveFactor)
	// Walk encoder-equivalent patches from reversed_decoder? Decoder starts after quantizer at lowest rate.
	// Match Python: after encoder, current_frame_rate is at bottleneck; decoder rebuilds from there.
	// Bottleneck rate = SR * interleave / product(encoder patches). Compute from decoder patches inverted.
	patchProd := 1
	for _, kw := range cfg.DecoderKwargs {
		if strFromAny(kw["module_type"]) == "PatchedPretransform" {
			patchProd *= intFromAny(kw["patch_size"], 1)
		}
	}
	if patchProd > 0 {
		frameRate = float64(cfg.SR()*interleaveFactor) / float64(patchProd)
	}

	var mods []decoderModule
	for _, kw := range cfg.DecoderKwargs {
		mt := strFromAny(kw["module_type"])
		switch mt {
		case "PatchedPretransform":
			ps := intFromAny(kw["patch_size"], 1)
			mods = append(mods, &patchedPretransform{patch: ps, downsample: false})
			frameRate *= float64(ps)
		case "Transformer":
			ctxDur := floatFromAny(kw["context_duration"], cfg.CausalCtxDuration)
			ctx := int(math.Round(frameRate * ctxDur))
			tm, err := loadProjectedTransformer(tensors, len(mods), kw, ctx)
			if err != nil {
				return nil, fmt.Errorf("decoder module %d: %w", len(mods), err)
			}
			mods = append(mods, tm)
		default:
			return nil, fmt.Errorf("unknown decoder module_type %q", mt)
		}
	}

	return &AudioTokenizer{
		Cfg: &cfg, Quant: quant, Decoder: mods,
		Channels: cfg.NumberChannels, Interleave: cfg.EnableChannelInterleave,
	}, nil
}

// SetFuse enables SIMD/GPU for decoder projected transformers.
func (at *AudioTokenizer) SetFuse(simdOn, gpuOn bool) {
	if at == nil {
		return
	}
	for _, m := range at.Decoder {
		if pt, ok := m.(*projectedTransformer); ok {
			pt.setFuse(simdOn, gpuOn)
		}
	}
}

// SyncGPU warms sticky FP32 decoder weights.
func (at *AudioTokenizer) SyncGPU() (int, error) {
	if at == nil {
		return 0, nil
	}
	n := 0
	for _, m := range at.Decoder {
		pt, ok := m.(*projectedTransformer)
		if !ok {
			continue
		}
		k, err := pt.warmGPU()
		n += k
		if err != nil {
			return n, err
		}
	}
	return n, nil
}

func findSTFile(dir string) (string, error) {
	for _, name := range []string{"model.safetensors", "model-00001-of-00001.safetensors"} {
		p := filepath.Join(dir, name)
		if st, err := os.Stat(p); err == nil && !st.IsDir() {
			return p, nil
		}
	}
	matches, _ := filepath.Glob(filepath.Join(dir, "model-*.safetensors"))
	if len(matches) > 0 {
		return matches[0], nil
	}
	return "", fmt.Errorf("no safetensors in %s", dir)
}

func intFromAny(v any, def int) int {
	switch t := v.(type) {
	case float64:
		return int(t)
	case int:
		return t
	case json.Number:
		i, _ := t.Int64()
		return int(i)
	default:
		return def
	}
}

func floatFromAny(v any, def float64) float64 {
	switch t := v.(type) {
	case float64:
		return t
	case int:
		return float64(t)
	default:
		return def
	}
}

func strFromAny(v any) string {
	s, _ := v.(string)
	return s
}

func boolFromAny(v any, def bool) bool {
	if v == nil {
		return def
	}
	b, ok := v.(bool)
	if ok {
		return b
	}
	return def
}

// DecodeCodes converts [nq][T] codes → interleaved float PCM (then restored to stereo).
// Returns planar interleaved stereo as [L0,R0,L1,R1,...] float32 in [-1,1], sample rate, channels.
func (at *AudioTokenizer) DecodeCodes(codes [][]int) ([]float32, int, int, error) {
	if at == nil || at.Quant == nil {
		return nil, 0, 0, fmt.Errorf("nil audio tokenizer")
	}
	nq := len(codes)
	if nq == 0 {
		return nil, 0, 0, fmt.Errorf("empty codes")
	}
	T := len(codes[0])
	for i := 1; i < nq; i++ {
		if len(codes[i]) != T {
			return nil, 0, 0, fmt.Errorf("ragged codes")
		}
	}
	// emb: (C=rvq→out_dim, T)
	emb := at.Quant.DecodeCodes(codes) // length outDim * T
	outDim := at.Quant.OutputDim
	x := emb
	ch, length := outDim, T
	for _, m := range at.Decoder {
		x, ch, length = m.Forward(x, ch, length)
	}
	// x is (1, length) channels=1 after final patch — actually channels = ch
	samples := x
	sr := at.Cfg.SR()
	channels := 1
	if at.Channels > 1 && at.Interleave {
		// restore: interleaved mono → stereo planar then to interleaved LRLR
		if length%at.Channels != 0 {
			length = (length / at.Channels) * at.Channels
			samples = samples[:length]
		}
		frames := length / at.Channels
		out := make([]float32, frames*at.Channels)
		for f := 0; f < frames; f++ {
			for c := 0; c < at.Channels; c++ {
				out[f*at.Channels+c] = samples[f*at.Channels+c]
			}
		}
		// Python: view (B, T, C).transpose → (B, C, T); our samples already (T*C) interleaved as T major with C last from view:
		// squeeze → (B, L) with L=T*C interleaved as [s0_ch0?]:
		// view(B, -1, C).transpose(1,2) means from interleaved [f0c0,f0c1,f1c0,f1c1] if stored that way.
		// From flatten: transpose(1,2).view → (B,1,C*T) with order t0c0,t0c1,t1c0,t1c1 for channels-last before view?
		// input (B,C,T).transpose(1,2) → (B,T,C).contiguous().view(B,1,-1) → order t0c0,t0c1,..., so interleaved by frame.
		// Restore: squeeze → (B, L), view(B, -1, C) → (B, T, C), transpose → (B, C, T).
		// So samples[f*C+c] is correct for interleaved WAV.
		samples = out
		channels = at.Channels
	} else {
		channels = ch
		if channels <= 0 {
			channels = 1
		}
	}
	return samples, sr, channels, nil
}

// DecodeFramesAR accepts AR frames [T][nvq] and decodes.
func (at *AudioTokenizer) DecodeFramesAR(frames [][]int) ([]float32, int, int, error) {
	if len(frames) == 0 {
		return nil, 0, 0, fmt.Errorf("no frames")
	}
	nvq := len(frames[0])
	T := len(frames)
	codes := make([][]int, nvq)
	for q := 0; q < nvq; q++ {
		codes[q] = make([]int, T)
		for t := 0; t < T; t++ {
			codes[q][t] = frames[t][q]
		}
	}
	return at.DecodeCodes(codes)
}
