package qwentts

import (
	"fmt"
	"math"
	"strings"
	"time"
)

// buildInputIDs wraps the text tokens with the fixed chat template:
//
//	<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
func (p *Pipeline) buildInputIDs(text string) (all, textToks []int) {
	textToks = p.tok.encode(text)
	all = append(all, tokIMStart, tokAssistant, tokNewline)
	all = append(all, textToks...)
	all = append(all, tokIMEnd, tokNewline, tokIMStart, tokAssistant, tokNewline)
	return all, textToks
}

// resolveLangSpk maps opts.Language/Speaker to codec vocab ids following the
// CustomVoice rules (Auto => no language token; dialect override for Chinese).
func (p *Pipeline) resolveLangSpk(opts SpeakOpts) (langID, spkID int, err error) {
	langID = -1
	lang := strings.ToLower(strings.TrimSpace(opts.Language))
	if lang == "" {
		lang = "auto"
	}
	if lang != "auto" {
		id, ok := p.cfg.LangID[lang]
		if !ok {
			return 0, 0, fmt.Errorf("qwentts: language %q not supported", opts.Language)
		}
		langID = id
	}
	spk := strings.ToLower(strings.TrimSpace(opts.Speaker))
	if spk == "" {
		return 0, 0, fmt.Errorf("qwentts: CustomVoice requires a speaker (e.g. \"Ryan\")")
	}
	id, ok := p.cfg.SpkID[spk]
	if !ok {
		return 0, 0, fmt.Errorf("qwentts: speaker %q not supported", opts.Speaker)
	}
	spkID = id
	// dialect override
	if lang == "chinese" || lang == "auto" {
		if d, ok := p.cfg.SpkIsDialect[spk]; ok && d != "" {
			if did, ok := p.cfg.LangID[strings.ToLower(d)]; ok {
				langID = did
			}
		}
	}
	return langID, spkID, nil
}

// generateCodes runs the CustomVoice generate loop and returns [T][16] codes.
//
// Matches the upstream default (non_streaming_mode=False): only the first text
// token is in the prefill; the remaining text tokens (+ tts_eos) are consumed
// one-per-frame as trailing_text_hidden during AR decode. Dumping all text into
// the prefill made the model emit codec_eos ~200 frames early on long passages.
func (p *Pipeline) generateCodes(text string, opts SpeakOpts) ([][]int, error) {
	t := p.talker
	H := t.cfg.HiddenSize

	langID, spkID, err := p.resolveLangSpk(opts)
	if err != nil {
		return nil, err
	}
	allIDs, textToks := p.buildInputIDs(text)
	if len(textToks) == 0 {
		return nil, fmt.Errorf("qwentts: empty text after tokenization")
	}
	_ = allIDs

	te := func(id int) []float32 { v := make([]float32, H); t.embedText(id, v); return v }
	ce := func(id int) []float32 { v := make([]float32, H); t.embedCodec(id, v); return v }
	addv := func(a, b []float32) []float32 {
		o := make([]float32, H)
		for i := 0; i < H; i++ {
			o[i] = a[i] + b[i]
		}
		return o
	}

	ttsPad := te(p.cfg.TTSPadID)
	ttsBos := te(p.cfg.TTSBosID)
	ttsEos := te(p.cfg.TTSEosID)

	// codec prefill tag ids (think / language / speaker / pad / bos)
	var cie0 []int
	if langID < 0 {
		cie0 = []int{p.cfg.CodecNoThink, p.cfg.CodecThinkBOS, p.cfg.CodecThinkEOS}
	} else {
		cie0 = []int{p.cfg.CodecThink, p.cfg.CodecThinkBOS, langID, p.cfg.CodecThinkEOS}
	}
	cieIDs := append([]int{}, cie0...)
	cieIDs = append(cieIDs, spkID)
	cieIDs = append(cieIDs, p.cfg.CodecPAD, p.cfg.CodecBOS)
	cieLen := len(cieIDs)

	var embeds []float32
	push := func(v []float32) { embeds = append(embeds, v...) }

	// role: <|im_start|>assistant\n
	for i := 0; i < 3; i++ {
		push(te(allIDs[i]))
	}
	// base: text=[pad*(cieLen-2), bos] + codec=cie[:cieLen-1]
	for j := 0; j < cieLen-1; j++ {
		txt := ttsPad
		if j == cieLen-2 {
			txt = ttsBos
		}
		push(addv(txt, ce(cieIDs[j])))
	}
	// first text token + codec_bos (cie last) — rest of text streams during decode
	push(addv(te(textToks[0]), ce(cieIDs[cieLen-1])))

	// trailing_text_hidden: text[1:] + tts_eos  (one token consumed per AR step)
	trailing := make([][]float32, 0, len(textToks))
	for i := 1; i < len(textToks); i++ {
		trailing = append(trailing, te(textToks[i]))
	}
	trailing = append(trailing, ttsEos)

	L := len(embeds) / H
	t.resetCache()
	lastHidden := t.forward(embeds, L)

	smp := newSampler(opts.Seed, opts)
	maxFrames := opts.MaxNewFrames
	if maxFrames <= 0 {
		maxFrames = 2048
	}

	vocab := t.codecHead.Out
	suppressStart := vocab - 1024
	if suppressStart < 0 {
		suppressStart = 0
	}
	logits := make([]float32, vocab)
	var frames [][]int
	var prev []int
	ce0 := make([]float32, H)
	emb := make([]float32, H)
	stoppedEOS := false
	t0 := time.Now()

	for step := 0; step < maxFrames; step++ {
		if step > 0 && step%10 == 0 {
			elapsed := time.Since(t0).Seconds()
			fps := float64(step) / elapsed
			audioSec := float64(step) / 12.0
			left := len(trailing) - step
			if left < 0 {
				left = 0
			}
			if left > 0 {
				fmt.Printf("  … frame %d · %.1fs audio · %.1f frame/s · text tokens left=%d\n", step, audioSec, fps, left)
			} else {
				// Text is fully conditioned; model keeps emitting speech frames on
				// tts_pad until it samples codec_eos (or hits max). Normal.
				fmt.Printf("  … frame %d · %.1fs audio · %.1f frame/s · text done, still speaking until eos\n", step, audioSec, fps)
			}
		}
		t.codecLogits(lastHidden, logits)
		for i := suppressStart; i < vocab; i++ {
			if i != p.cfg.CodecEOS {
				logits[i] = float32(math.Inf(-1))
			}
		}
		if step < 2 { // min_new_tokens (upstream default)
			logits[p.cfg.CodecEOS] = float32(math.Inf(-1))
		}
		code0 := smp.sampleCodec(logits, prev)
		if code0 == p.cfg.CodecEOS {
			stoppedEOS = true
			break
		}
		prev = append(prev, code0)

		t.embedCodec(code0, ce0)
		rest := p.codePred.predict(lastHidden, ce0, smp)
		frame := make([]int, 0, 1+len(rest))
		frame = append(frame, code0)
		frame = append(frame, rest...)
		frames = append(frames, frame)

		// next talker input: sum(ce(code0), group_i(rest_i)) + trailing_text[step] | tts_pad
		sum := make([]float32, H)
		copy(sum, ce0)
		for i := 0; i < len(rest); i++ {
			p.codePred.groupEmbed(i, rest[i], emb)
			for d := 0; d < H; d++ {
				sum[d] += emb[d]
			}
		}
		var textAdd []float32
		if step < len(trailing) {
			textAdd = trailing[step]
		} else {
			textAdd = ttsPad
		}
		for d := 0; d < H; d++ {
			sum[d] += textAdd[d]
		}
		lastHidden = t.forward(sum, 1)
	}

	if len(frames) == 0 {
		return nil, fmt.Errorf("qwentts: no audio frames generated")
	}
	if stoppedEOS {
		elapsed := time.Since(t0).Seconds()
		fps := float64(len(frames)) / elapsed
		fmt.Printf("  … stopped at codec_eos after %d frames in %.1fs (%.1f frame/s, max=%d)\n", len(frames), elapsed, fps, maxFrames)
	} else {
		elapsed := time.Since(t0).Seconds()
		fps := float64(len(frames)) / elapsed
		fmt.Printf("  … hit max_frames=%d in %.1fs (%.1f frame/s, no codec_eos)\n", maxFrames, elapsed, fps)
	}
	return frames, nil
}
