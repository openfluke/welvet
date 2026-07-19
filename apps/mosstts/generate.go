package mosstts

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
)

// GenOpts controls AR sampling.
type GenOpts struct {
	MaxNewFrames     int
	DoSample         bool
	TextTemperature  float32
	TextTopK         int
	TextTopP         float32
	AudioTemperature float32
	AudioTopK        int
	AudioTopP        float32
	Seed             int64
	NQ               int
}

func defaultGenOpts(o GenOpts) GenOpts {
	if o.MaxNewFrames <= 0 {
		o.MaxNewFrames = 300
	}
	if o.TextTemperature <= 0 {
		o.TextTemperature = 1.5
	}
	if o.AudioTemperature <= 0 {
		o.AudioTemperature = 1.7
	}
	if o.TextTopK <= 0 {
		o.TextTopK = 50
	}
	if o.AudioTopK <= 0 {
		o.AudioTopK = 25
	}
	if o.TextTopP <= 0 {
		o.TextTopP = 1
	}
	if o.AudioTopP <= 0 {
		o.AudioTopP = 0.8
	}
	return o
}

// GenerateFrames runs global+local AR and returns audio codes [frames][nvq].
func (m *Model) GenerateFrames(promptRows [][]int, opts GenOpts) ([][]int, error) {
	opts = defaultGenOpts(opts)
	cfg := m.Cfg
	nq := opts.NQ
	if nq <= 0 || nq > cfg.NVQ {
		nq = cfg.NVQ
	}
	h := cfg.HiddenSize
	rng := rand.New(rand.NewSource(opts.Seed))

	seq := len(promptRows)
	if seq == 0 {
		return nil, fmt.Errorf("empty prompt")
	}
	e := m.buildInputsEmbeds(promptRows)
	mask := make([]bool, 0, seq+opts.MaxNewFrames)
	var lastHidden []float32

	useGPU := m.useFuseGPU && m.fuseGlobal != nil && m.fuseLocal != nil
	var caches []kvCache
	if useGPU {
		if err := m.fuseGlobal.ResetPos(); err != nil {
			return nil, err
		}
		for pos := 0; pos < seq; pos++ {
			tok := make([]float32, h)
			copy(tok, e[pos*h:(pos+1)*h])
			mask = append(mask, true)
			if err := m.fuseGlobal.DecodeStepFused(tok); err != nil {
				return nil, fmt.Errorf("global fuse prefill: %w", err)
			}
			lastHidden = tok
		}
	} else {
		caches = make([]kvCache, len(m.Global.Blocks))
		for pos := 0; pos < seq; pos++ {
			tok := make([]float32, h)
			copy(tok, e[pos*h:(pos+1)*h])
			mask = append(mask, true)
			m.Global.ForwardLast(tok, caches, mask)
			lastHidden = tok
		}
	}

	var frames [][]int
	for step := 0; step < opts.MaxNewFrames; step++ {
		if step == 0 || (step+1)%10 == 0 || step+1 == opts.MaxNewFrames {
			fmt.Printf("\r  AR frame %d/%d", step+1, opts.MaxNewFrames)
		}

		var hidden []float32
		if useGPU {
			if err := m.fuseLocal.ResetPos(); err != nil {
				return nil, err
			}
			tok := make([]float32, h)
			copy(tok, lastHidden)
			if err := m.fuseLocal.DecodeStepFused(tok); err != nil {
				return nil, fmt.Errorf("local fuse: %w", err)
			}
			hidden = tok
		} else {
			localCaches := make([]kvCache, len(m.Local.Blocks))
			tok := make([]float32, h)
			copy(tok, lastHidden)
			m.Local.DecodeStep(tok, localCaches)
			hn := make([]float32, h)
			copy(hn, tok)
			m.Local.FinalNorm(hn)
			hidden = hn

			textLogits := make([]float32, cfg.VocabSize)
			m.TextHead.Forward(hidden, textLogits)
			nextText := sampleAssistantTextToken(textLogits, cfg.AudioAssistantSlotID, cfg.AudioEndTokenID,
				opts.DoSample, opts.TextTemperature, opts.TextTopK, opts.TextTopP, rng)

			if nextText == cfg.AudioEndTokenID {
				break
			}
			if nextText != cfg.AudioAssistantSlotID {
				break
			}

			frame := make([]int, cfg.NVQ)
			for i := range frame {
				frame[i] = cfg.AudioPadTokenID
			}
			textEmb := make([]float32, h)
			m.Global.embedToken(nextText, textEmb)
			copy(tok, textEmb)
			m.Local.DecodeStep(tok, localCaches)

			for ch := 0; ch < nq; ch++ {
				hn := make([]float32, h)
				copy(hn, tok)
				m.Local.FinalNorm(hn)
				logits := make([]float32, cfg.AudioCodebookSizes[ch])
				m.audioLMHead(ch, hn, logits)
				tokID := sampleLogits(logits, opts.DoSample, opts.AudioTemperature, opts.AudioTopK, opts.AudioTopP, rng)
				frame[ch] = tokID
				aemb := make([]float32, h)
				copy(aemb, m.AudioEmb[ch][tokID*h:(tokID+1)*h])
				copy(tok, aemb)
				m.Local.DecodeStep(tok, localCaches)
			}
			frames = append(frames, frame)

			row := make([]int, 1+cfg.NVQ)
			row[0] = cfg.AudioAssistantSlotID
			copy(row[1:], frame)
			emb := m.buildInputsEmbeds([][]int{row})
			gtok := make([]float32, h)
			copy(gtok, emb)
			mask = append(mask, true)
			m.Global.ForwardLast(gtok, caches, mask)
			lastHidden = gtok
			continue
		}

		// GPU local path
		textLogits := make([]float32, cfg.VocabSize)
		m.TextHead.Forward(hidden, textLogits)
		nextText := sampleAssistantTextToken(textLogits, cfg.AudioAssistantSlotID, cfg.AudioEndTokenID,
			opts.DoSample, opts.TextTemperature, opts.TextTopK, opts.TextTopP, rng)

		if nextText == cfg.AudioEndTokenID {
			break
		}
		if nextText != cfg.AudioAssistantSlotID {
			break
		}

		frame := make([]int, cfg.NVQ)
		for i := range frame {
			frame[i] = cfg.AudioPadTokenID
		}
		textEmb := make([]float32, h)
		m.Global.embedToken(nextText, textEmb)
		tok := make([]float32, h)
		copy(tok, textEmb)
		if err := m.fuseLocal.DecodeStepNoFinalLN(tok); err != nil {
			return nil, err
		}

		for ch := 0; ch < nq; ch++ {
			hn := make([]float32, h)
			copy(hn, tok)
			if err := m.fuseLocal.FinalNormGPU(hn); err != nil {
				return nil, err
			}
			logits := make([]float32, cfg.AudioCodebookSizes[ch])
			m.audioLMHead(ch, hn, logits)
			tokID := sampleLogits(logits, opts.DoSample, opts.AudioTemperature, opts.AudioTopK, opts.AudioTopP, rng)
			frame[ch] = tokID
			aemb := make([]float32, h)
			copy(aemb, m.AudioEmb[ch][tokID*h:(tokID+1)*h])
			copy(tok, aemb)
			if err := m.fuseLocal.DecodeStepNoFinalLN(tok); err != nil {
				return nil, err
			}
		}
		frames = append(frames, frame)

		row := make([]int, 1+cfg.NVQ)
		row[0] = cfg.AudioAssistantSlotID
		copy(row[1:], frame)
		emb := m.buildInputsEmbeds([][]int{row})
		gtok := make([]float32, h)
		copy(gtok, emb)
		mask = append(mask, true)
		if err := m.fuseGlobal.DecodeStepFused(gtok); err != nil {
			return nil, fmt.Errorf("global fuse decode: %w", err)
		}
		lastHidden = gtok
	}
	if len(frames) > 0 {
		fmt.Printf("\r  AR frames %d (done)          \n", len(frames))
	} else {
		fmt.Printf("\r  AR produced 0 frames          \n")
	}
	return frames, nil
}

func sampleAssistantTextToken(logits []float32, slotID, endID int, doSample bool, temp float32, topK int, topP float32, rng *rand.Rand) int {
	cands := []int{slotID, endID}
	candLogits := make([]float32, len(cands))
	for i, id := range cands {
		if id >= 0 && id < len(logits) {
			candLogits[i] = logits[id]
		} else {
			candLogits[i] = float32(-1e30)
		}
	}
	idx := sampleLogits(candLogits, doSample, temp, topK, topP, rng)
	if idx < 0 || idx >= len(cands) {
		return endID
	}
	return cands[idx]
}

func sampleLogits(logits []float32, doSample bool, temp float32, topK int, topP float32, rng *rand.Rand) int {
	if !doSample || temp <= 0 {
		best := 0
		for i := 1; i < len(logits); i++ {
			if logits[i] > logits[best] {
				best = i
			}
		}
		return best
	}
	scaled := make([]float32, len(logits))
	for i, v := range logits {
		scaled[i] = v / temp
	}
	if topK > 0 && topK < len(scaled) {
		idx := argsortDesc(scaled)
		thresh := scaled[idx[topK-1]]
		for i := range scaled {
			if scaled[i] < thresh {
				scaled[i] = float32(-1e30)
			}
		}
	}
	maxV := scaled[0]
	for _, v := range scaled[1:] {
		if v > maxV {
			maxV = v
		}
	}
	var sum float64
	probs := make([]float64, len(scaled))
	for i, v := range scaled {
		e := math.Exp(float64(v - maxV))
		probs[i] = e
		sum += e
	}
	for i := range probs {
		probs[i] /= sum
	}
	if topP > 0 && topP < 1 {
		idx := argsortDesc(scaled)
		var cum float64
		allowed := make([]bool, len(probs))
		for _, i := range idx {
			allowed[i] = true
			cum += probs[i]
			if cum >= float64(topP) {
				break
			}
		}
		var sum2 float64
		for i := range probs {
			if !allowed[i] {
				probs[i] = 0
			}
			sum2 += probs[i]
		}
		if sum2 > 0 {
			for i := range probs {
				probs[i] /= sum2
			}
		}
	}
	r := rng.Float64()
	var cum float64
	for i, p := range probs {
		cum += p
		if r <= cum {
			return i
		}
	}
	return len(probs) - 1
}

func argsortDesc(a []float32) []int {
	idx := make([]int, len(a))
	for i := range idx {
		idx[i] = i
	}
	for i := 0; i < len(idx); i++ {
		best := i
		for j := i + 1; j < len(idx); j++ {
			if a[idx[j]] > a[idx[best]] {
				best = j
			}
		}
		idx[i], idx[best] = idx[best], idx[i]
	}
	return idx
}

// BuildTextPromptRows builds [seq][1+nvq] prompt without reference audio.
func BuildTextPromptRows(sp *SentencePiece, cfg *Config, text string) ([][]int, error) {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil, fmt.Errorf("empty text")
	}
	textIDs := sp.Encode(text)
	prefix := buildPromptPrefixIDs(sp, cfg)
	suffix := buildPromptSuffixIDs(sp, cfg)
	ids := append(append(prefix, textIDs...), suffix...)
	rows := make([][]int, len(ids))
	for i, id := range ids {
		row := make([]int, 1+cfg.NVQ)
		row[0] = id
		for c := 0; c < cfg.NVQ; c++ {
			row[1+c] = cfg.AudioPadTokenID
		}
		rows[i] = row
	}
	return rows, nil
}

func spEncodeLit(sp *SentencePiece, s string) []int {
	// Match upstream prompting.encode_text → tokenizer.encode(text).
	return sp.Encode(s)
}

func buildPromptPrefixIDs(sp *SentencePiece, cfg *Config) []int {
	out := []int{cfg.ImStartTokenID}
	out = append(out, spEncodeLit(sp, "user\n")...)
	out = append(out, spEncodeLit(sp, "<user_inst>\n- Reference(s):\n")...)
	out = append(out, spEncodeLit(sp, "None")...)
	out = append(out, spEncodeLit(sp, "\n- Instruction:\nNone\n- Tokens:\nNone\n- Quality:\nNone\n- Sound Event:\nNone\n- Ambient Sound:\nNone\n- Language:\nNone\n- Text:\n")...)
	return out
}

func buildPromptSuffixIDs(sp *SentencePiece, cfg *Config) []int {
	out := spEncodeLit(sp, "\n</user_inst>")
	out = append(out, cfg.ImEndTokenID)
	out = append(out, spEncodeLit(sp, "\n")...)
	out = append(out, cfg.ImStartTokenID)
	out = append(out, spEncodeLit(sp, "assistant\n")...)
	return out
}
