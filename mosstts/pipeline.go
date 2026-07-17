package mosstts

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/openfluke/welvet/webgpu"
)

// SpeakOpts controls end-to-end TTS.
type SpeakOpts struct {
	MaxNewFrames int
	DoSample     bool
	Seed         int64
	NQ           int
	RefWAV       string // optional reference for clone (MVP: ignored unless encode path added)
	FuseSIMD     bool
	FuseGPU      bool
}

// Pipeline holds AR + codec + SP for one snapshot pair.
type Pipeline struct {
	AR     *Model
	Codec  *AudioTokenizer
	SP     *SentencePiece
	TTSDir string
}

// LoadPipeline loads TTS AR snapshot and neighboring audio tokenizer.
func LoadPipeline(ttsSnap string) (*Pipeline, error) {
	ar, err := LoadAR(ttsSnap)
	if err != nil {
		return nil, err
	}
	spPath := filepath.Join(ttsSnap, "tokenizer.model")
	if _, err := os.Stat(spPath); err != nil {
		return nil, fmt.Errorf("mosstts: need tokenizer.model: %w", err)
	}
	sp, err := LoadTokenizerModel(spPath)
	if err != nil {
		return nil, err
	}
	atDir, err := FindAudioTokenizerDir(ttsSnap)
	if err != nil {
		return nil, err
	}
	codec, err := LoadAudioTokenizer(atDir)
	if err != nil {
		return nil, fmt.Errorf("audio tokenizer: %w", err)
	}
	return &Pipeline{AR: ar, Codec: codec, SP: sp, TTSDir: ttsSnap}, nil
}

// ApplyFuse configures SIMD/GPU backends and optionally warms VRAM.
func (p *Pipeline) ApplyFuse(simdOn, gpuOn bool) error {
	if p == nil {
		return fmt.Errorf("nil pipeline")
	}
	if gpuOn {
		simdOn = true // SIMD for attention dots / host fallback
		if !webgpu.Available() {
			err := webgpu.InitError()
			if err == nil {
				err = fmt.Errorf("no adapter")
			}
			return fmt.Errorf("GPU fuse: %w", err)
		}
	}
	p.AR.SetFuse(simdOn, gpuOn)
	p.Codec.SetFuse(simdOn, gpuOn)
	if !gpuOn {
		return nil
	}
	webgpu.ClearF32WeightCache()
	n1, err := p.AR.SyncGPU()
	if err != nil {
		return err
	}
	n2, err := p.Codec.SyncGPU()
	if err != nil {
		return err
	}
	fmt.Printf("  moss GPU fuse: resident GPT-2 decode (%d AR mats warmed; codec sticky=%d)\n", n1, n2)
	return nil
}

// CloseGPU drops sticky FP32 weights.
func (p *Pipeline) CloseGPU() {
	webgpu.ClearF32WeightCache()
}

// Speak synthesizes text → samples.
func (p *Pipeline) Speak(text string, opts SpeakOpts) (samples []float32, sampleRate, channels int, err error) {
	if p == nil || p.AR == nil || p.Codec == nil || p.SP == nil {
		return nil, 0, 0, fmt.Errorf("nil pipeline")
	}
	_ = opts.RefWAV
	if err := p.ApplyFuse(opts.FuseSIMD, opts.FuseGPU); err != nil {
		return nil, 0, 0, err
	}
	if opts.FuseGPU {
		defer p.CloseGPU()
	}
	rows, err := BuildTextPromptRows(p.SP, p.AR.Cfg, text)
	if err != nil {
		return nil, 0, 0, err
	}
	frames, err := p.AR.GenerateFrames(rows, GenOpts{
		MaxNewFrames: opts.MaxNewFrames,
		DoSample:     opts.DoSample,
		Seed:         opts.Seed,
		NQ:           opts.NQ,
	})
	if err != nil {
		return nil, 0, 0, err
	}
	if len(frames) == 0 {
		return nil, 0, 0, fmt.Errorf("no audio frames generated")
	}
	return p.Codec.DecodeFramesAR(frames)
}

// SpeakToFile runs Speak and writes a WAV under outDir.
func (p *Pipeline) SpeakToFile(text, outDir string, opts SpeakOpts) (string, error) {
	samples, sr, ch, err := p.Speak(text, opts)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return "", err
	}
	name := fmt.Sprintf("moss_%s.wav", time.Now().Format("20060102-150405"))
	path := filepath.Join(outDir, name)
	if err := WriteWAV(path, samples, sr, ch); err != nil {
		return "", err
	}
	_ = os.WriteFile(path+".txt", []byte(text+"\n"), 0o644)
	return path, nil
}
