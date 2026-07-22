package qwenasr

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/openfluke/welvet/model/wav2vec2"
	"github.com/openfluke/welvet/simd"
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
	fuse := opts.FuseSIMD
	if fuse && !simd.Enabled() {
		fuse = false
	}
	p.enc.SetSIMD(fuse)
	p.dec.SetSIMD(fuse)

	tAll := time.Now()
	mel := melSpectrogram(pcm)
	tMel := time.Since(tAll)
	t0 := time.Now()
	audio := p.enc.forward(mel)
	tEnc := time.Since(t0)
	nAudio := len(audio) / p.cfg.Decoder.Hidden
	ids := append(append([]int{}, promptPrefix...), make([]int, nAudio)...)
	for i := len(promptPrefix); i < len(promptPrefix)+nAudio; i++ {
		ids[i] = audioTokenID
	}
	ids = append(ids, promptSuffix...)

	H := p.cfg.Decoder.Hidden
	embeds := make([]float32, len(ids)*H)
	for i, id := range ids {
		dst := embeds[i*H : (i+1)*H]
		if id == audioTokenID {
			copy(dst, audio[(i-len(promptPrefix))*H:(i-len(promptPrefix)+1)*H])
		} else {
			p.dec.embedInto(id, dst)
		}
	}

	p.dec.reset()
	t0 = time.Now()
	h := p.dec.forward(embeds, len(ids))
	prefillMS := time.Since(t0)

	max := opts.MaxNewTokens
	if max <= 0 {
		max = 256
	}
	out := make([]int, 0, 64)
	tokEmb := make([]float32, H)
	t0 = time.Now()
	for i := 0; i < max; i++ {
		logits := p.dec.logits(h)
		best := argmax(logits)
		out = append(out, best)
		if best == eosID || best == imEndID {
			break
		}
		p.dec.embedInto(best, tokEmb)
		h = p.dec.forward(tokEmb, 1)
	}
	decodeMS := time.Since(t0)
	fmt.Fprintf(os.Stderr, "  qwenasr: mel=%v enc=%v audio_tok=%d prefill=%v dec_tok=%d decode=%v simd=%v\n",
		tMel.Round(time.Millisecond), tEnc.Round(time.Millisecond), nAudio,
		prefillMS.Round(time.Millisecond), len(out), decodeMS.Round(time.Millisecond), fuse)
	text := p.tok.decode(out)
	if x := strings.Index(text, "<asr_text>"); x >= 0 {
		text = text[x+len("<asr_text>"):]
	}
	return strings.TrimSpace(text), nil
}

func argmax(x []float32) int {
	best := 0
	// Parallel reduce over large vocab (lm_head ~152k).
	workers := runtime.GOMAXPROCS(0)
	if workers < 2 || len(x) < 4096 {
		for j := 1; j < len(x); j++ {
			if x[j] > x[best] {
				best = j
			}
		}
		return best
	}
	type pair struct{ i int; v float32 }
	parts := make([]pair, workers)
	var wg sync.WaitGroup
	chunk := (len(x) + workers - 1) / workers
	for w := 0; w < workers; w++ {
		lo := w * chunk
		hi := lo + chunk
		if hi > len(x) {
			hi = len(x)
		}
		if lo >= hi {
			parts[w] = pair{0, float32(math.Inf(-1))}
			continue
		}
		wg.Add(1)
		go func(w, a, b int) {
			defer wg.Done()
			bi := a
			bv := x[a]
			for j := a + 1; j < b; j++ {
				if x[j] > bv {
					bi, bv = j, x[j]
				}
			}
			parts[w] = pair{bi, bv}
		}(w, lo, hi)
	}
	wg.Wait()
	best = parts[0].i
	bv := parts[0].v
	for _, p := range parts[1:] {
		if p.v > bv {
			best, bv = p.i, p.v
		}
	}
	return best
}
