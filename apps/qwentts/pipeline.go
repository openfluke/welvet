package qwentts

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/openfluke/welvet/simd"
	"github.com/openfluke/welvet/webgpu"
)

// SpeakOpts controls end-to-end CustomVoice synthesis.
type SpeakOpts struct {
	MaxNewFrames int
	DoSample     bool
	Seed         int64
	Speaker      string // e.g. "Ryan"
	Language     string // e.g. "English" or "Auto"
	Instruct     string
	RefWAV       string
	FuseSIMD     bool
	FuseGPU      bool
}

// Pipeline holds the parsed config + talker + code predictor + speech decoder
// for one Qwen3-TTS-12Hz CustomVoice snapshot.
type Pipeline struct {
	cfg      *Config
	tok      *bpeTokenizer
	talker   *Talker
	codePred *CodePredictor
	decoder  *Decoder
	SnapDir  string

	fuseSIMD bool
	fuseGPU  bool
}

// LoadPipeline loads a Qwen3-TTS CustomVoice snapshot (talker + speech
// tokenizer decoder + tokenizer). It performs no download.
func LoadPipeline(ttsSnap string) (*Pipeline, error) {
	if fi, err := os.Stat(ttsSnap); err != nil || !fi.IsDir() {
		return nil, fmt.Errorf("qwentts: snapshot dir not found: %s", ttsSnap)
	}
	cfg, err := LoadConfig(ttsSnap)
	if err != nil {
		return nil, err
	}
	tok, err := loadBPETokenizer(ttsSnap)
	if err != nil {
		return nil, err
	}
	talker, err := loadTalker(ttsSnap, cfg.Talker)
	if err != nil {
		return nil, fmt.Errorf("qwentts talker: %w", err)
	}
	numGroups := cfg.Talker.NumCodeGroups - 1
	codePred, err := loadCodePredictor(ttsSnap, cfg.CodePredictor, numGroups)
	if err != nil {
		return nil, fmt.Errorf("qwentts code_predictor: %w", err)
	}
	decoder, err := loadDecoder(ttsSnap, cfg.Decoder)
	if err != nil {
		return nil, fmt.Errorf("qwentts decoder: %w", err)
	}
	return &Pipeline{
		cfg:      cfg,
		tok:      tok,
		talker:   talker,
		codePred: codePred,
		decoder:  decoder,
		SnapDir:  ttsSnap,
	}, nil
}

func (p *Pipeline) warmSoft(label string, fn func() (int, error)) error {
	n, err := fn()
	if err == nil {
		if n > 0 {
			fmt.Printf("  qwen GPU warm %s: %d mats\n", label, n)
		}
		return nil
	}
	if webgpu.IsF32VRAMFull(err) {
		fmt.Printf("  qwen GPU warm %s: VRAM soft-cap — host/SIMD for rest\n", label)
		return nil
	}
	return err
}

// ApplyFuse configures SIMD + talker resident GPU fuse for the AR stage.
// Code predictor stays on SIMD during AR (sticky per-GEMV there is ~100 syncs
// per frame and makes GPU fuse slower than CPU). Decoder is warmed later
// after the talker fuse is freed (see armDecoderGPU).
func (p *Pipeline) ApplyFuse(simdOn, gpuOn bool) error {
	if p == nil {
		return fmt.Errorf("nil pipeline")
	}
	p.fuseSIMD, p.fuseGPU = simdOn, gpuOn
	if gpuOn {
		simdOn = true // host/SIMD fallback for heads + code predictor
		p.fuseSIMD = true
		if !webgpu.Available() {
			err := webgpu.InitError()
			if err == nil {
				err = fmt.Errorf("no adapter")
			}
			return fmt.Errorf("GPU fuse: %w", err)
		}
	}
	if simdOn && !simd.Enabled() {
		fmt.Println("  (SIMD kernels unavailable on this GOARCH — scalar host)")
		simdOn = false
		p.fuseSIMD = false
	}
	p.talker.SetFuse(simdOn, gpuOn)
	// CP: always SIMD alongside talker fuse (or SIMD-only when no GPU).
	p.codePred.SetFuse(simdOn, false)
	p.decoder.SetFuse(simdOn, false)
	if !gpuOn {
		return nil
	}
	webgpu.ClearF32WeightCache()
	if ms := p.talker.FuseMaxSeq(); ms > 0 {
		fmt.Printf("  qwen GPU fuse: talker decode (1 submit/token, maxSeq=%d); CP/heads=SIMD; decoder later\n", ms)
	} else {
		fmt.Println("  qwen GPU fuse: talker fuse unavailable — SIMD AR")
	}
	return nil
}

// armDecoderGPU frees the talker fuse / sticky cache before vocoding.
// Decoder linears stay on SIMD — sticky per-GEMV WebGPU here is far slower
// (thousands of syncs across T timesteps + ConvNeXt).
func (p *Pipeline) armDecoderGPU() error {
	if p == nil || !p.fuseGPU {
		return nil
	}
	p.talker.CloseFuse()
	webgpu.ClearF32WeightCache()
	p.decoder.SetFuse(p.fuseSIMD, false)
	fmt.Println("  qwen decode: vocoder on SIMD (talker VRAM freed)")
	return nil
}

// CloseGPU tears down the talker decode fuse and drops sticky FP32 weights.
func (p *Pipeline) CloseGPU() {
	if p != nil && p.talker != nil {
		p.talker.CloseFuse()
	}
	webgpu.ClearF32WeightCache()
}

// Speak synthesizes text into mono 24 kHz float32 PCM.
func (p *Pipeline) Speak(text string, opts SpeakOpts) (samples []float32, sampleRate, channels int, err error) {
	if p == nil || p.talker == nil || p.decoder == nil {
		return nil, 0, 0, fmt.Errorf("qwentts: nil pipeline")
	}
	if err := p.ApplyFuse(opts.FuseSIMD, opts.FuseGPU); err != nil {
		p.CloseGPU() // drop partial fuse / sticky warm on setup failure
		return nil, 0, 0, err
	}
	if opts.FuseGPU {
		defer p.CloseGPU()
	}
	frames, err := p.generateCodes(text, opts)
	if err != nil {
		return nil, 0, 0, err
	}
	if err := p.armDecoderGPU(); err != nil {
		return nil, 0, 0, err
	}
	samples, err = p.decoder.decode(frames)
	if err != nil {
		return nil, 0, 0, err
	}
	return samples, p.cfg.Decoder.SampleRate, 1, nil
}

// SpeakToFile runs Speak and writes a WAV under outDir (qwen_<timestamp>.wav).
func (p *Pipeline) SpeakToFile(text, outDir string, opts SpeakOpts) (string, error) {
	samples, sr, ch, err := p.Speak(text, opts)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return "", err
	}
	name := fmt.Sprintf("qwen_%s.wav", time.Now().Format("20060102-150405"))
	path := filepath.Join(outDir, name)
	if err := WriteWAV(path, samples, sr, ch); err != nil {
		return "", err
	}
	_ = os.WriteFile(path+".txt", []byte(text+"\n"), 0o644)
	return path, nil
}
