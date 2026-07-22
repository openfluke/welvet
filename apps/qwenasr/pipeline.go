package qwenasr

import (
	"fmt"
	"github.com/openfluke/welvet/model/wav2vec2"
	"strings"
)

type TranscribeOpts struct {
	MaxNewTokens int
	FuseSIMD     bool
}
type Pipeline struct {
	cfg *Config
	enc *encoder
	dec *decoder
	tok *bpeTokenizer
}

func LoadPipeline(snap string) (*Pipeline, error) {
	c, e := LoadConfig(snap)
	if e != nil {
		return nil, e
	}
	s, e := openTensorStore(snap)
	if e != nil {
		return nil, e
	}
	en, e := newEncoder(s, c.Encoder)
	if e != nil {
		return nil, fmt.Errorf("qwenasr encoder: %w", e)
	}
	d, e := newDecoder(s, c.Decoder)
	if e != nil {
		return nil, fmt.Errorf("qwenasr decoder: %w", e)
	}
	t, e := loadTokenizer(snap)
	if e != nil {
		return nil, fmt.Errorf("qwenasr tokenizer: %w", e)
	}
	return &Pipeline{c, en, d, t}, nil
}
func (p *Pipeline) TranscribeFile(path string, opts TranscribeOpts) (string, error) {
	x, sr, e := wav2vec2.ReadWAVMono(path)
	if e != nil {
		return "", e
	}
	return p.TranscribePCM(x, sr, opts)
}
func (p *Pipeline) TranscribePCM(pcm []float32, sr int, opts TranscribeOpts) (string, error) {
	if sr <= 0 {
		return "", fmt.Errorf("qwenasr: invalid sample rate %d", sr)
	}
	if sr != 16000 {
		pcm = wav2vec2.ResampleLinear(pcm, sr, 16000)
	}
	if len(pcm) == 0 {
		return "", fmt.Errorf("qwenasr: empty audio")
	}
	p.enc.SetSIMD(opts.FuseSIMD)
	p.dec.SetSIMD(opts.FuseSIMD)
	audio := p.enc.forward(melSpectrogram(pcm))
	n := len(audio) / p.cfg.Decoder.Hidden
	ids := append(append([]int{}, promptPrefix...), make([]int, n)...)
	for i := len(promptPrefix); i < len(promptPrefix)+n; i++ {
		ids[i] = audioTokenID
	}
	ids = append(ids, promptSuffix...)
	p.dec.reset()
	var h []float32
	for i, id := range ids {
		x := p.dec.embed(id)
		if id == audioTokenID {
			copy(x, audio[(i-len(promptPrefix))*p.cfg.Decoder.Hidden:])
		}
		h = p.dec.step(x, i)
	}
	max := opts.MaxNewTokens
	if max <= 0 {
		max = 256
	}
	out := []int{}
	for i := 0; i < max; i++ {
		logits := p.dec.logits(h)
		best := 0
		for j := 1; j < len(logits); j++ {
			if logits[j] > logits[best] {
				best = j
			}
		}
		out = append(out, best)
		if best == eosID || best == imEndID {
			break
		}
		h = p.dec.step(p.dec.embed(best), len(ids)+i)
	}
	text := p.tok.decode(out)
	if x := strings.Index(text, "<asr_text>"); x >= 0 {
		text = text[x+len("<asr_text>"):]
	}
	return strings.TrimSpace(text), nil
}
