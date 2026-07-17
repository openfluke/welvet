package transformer

import (
	"fmt"
	"math"
	"time"

	"github.com/openfluke/welvet/fusedgpu"
	"github.com/openfluke/welvet/sampling"
)

// GenOptions controls greedy generation + streaming.
type GenOptions struct {
	MaxTokens int
	// Silent suppresses stdout streaming (metrics footer still prints unless PrintMetrics is false).
	Silent bool
	// PrintMetrics prints the Lucy-style tok/s footer when generation produced tokens.
	PrintMetrics bool
	// StreamCallback receives each streamed text chunk (after ChatML cleanup).
	StreamCallback func(piece string)
	// RepetitionPenalty > 1 down-weights recent tokens (0 = default 1.15; <0 disables).
	RepetitionPenalty float32
	// RepetitionWindow is how many recent tokens the penalty considers (0 = 64).
	RepetitionWindow int
	// NoRepeatNGram stops when the last n tokens repeat immediately (0 = 8; <0 disables).
	NoRepeatNGram int
}

// Generate runs greedy decode for one user message (ChatML by default).
// Streams tokens to stdout unless Silent. Returns metrics like Lucy ENTITY Talk.
func (m *Model) Generate(
	encode func(text string, addSpecial bool) []uint32,
	decode func(ids []uint32, skipSpecial bool) string,
	turns []Turn,
	systemPrompt, userMsg string,
	opts GenOptions,
) (string, GenMetrics, error) {
	var zero GenMetrics
	if m == nil {
		return "", zero, fmt.Errorf("transformer: nil model")
	}
	if opts.MaxTokens <= 0 {
		opts.MaxTokens = 1024
	}
	if !opts.Silent && !opts.PrintMetrics {
		opts.PrintMetrics = true
	}
	if opts.RepetitionPenalty == 0 {
		opts.RepetitionPenalty = 1.15
	}
	if opts.RepetitionWindow <= 0 {
		opts.RepetitionWindow = 64
	}
	if opts.NoRepeatNGram == 0 {
		opts.NoRepeatNGram = 8
	}

	prompt := ChatML.BuildPrompt(turns, systemPrompt, userMsg)
	// Qwen3 / Bonsai: hard-disable thinking via empty <think></think> in the prompt.
	if m.Architecture == "qwen3_dense" || m.Architecture == "qwen35_hybrid" {
		prompt = ChatML.BuildPromptNoThink(turns, systemPrompt, userMsg)
	}
	ids := encode(prompt, false)
	if len(ids) == 0 {
		return "", zero, fmt.Errorf("tokenizer produced empty prompt")
	}

	streamDecode := func(toks []uint32) string { return decode(toks, false) }
	allTokens := append([]uint32(nil), ids...)
	stream := NewStreamer(streamDecode, ids)

	eos := make(map[int]struct{}, len(m.EOSTokens))
	for _, t := range m.EOSTokens {
		eos[t] = struct{}{}
	}

	m.ResetKV()
	m.Quiet = opts.Silent
	prefillStart := time.Now()

	// Hybrid full-fuse: on-device argmax (map 4 bytes/token), not full logits.
	if he, ok := m.gpu.(*fusedgpu.HybridEngine); ok && he != nil {
		return m.generateHybridGPUSample(he, encode, decode, ids, eos, stream, allTokens, opts, prefillStart)
	}

	logits, err := m.ForwardTokens(ids)
	m.Quiet = false
	prefillElapsed := time.Since(prefillStart)
	if err != nil {
		return "", zero, fmt.Errorf("prefill: %w", err)
	}
	if !opts.Silent {
		fmt.Printf("  prompt loaded in %s (%.2f tok/s)\nAssistant: ",
			prefillElapsed.Round(time.Millisecond),
			float64(len(ids))/math.Max(prefillElapsed.Seconds(), 1e-9))
	}

	decodeStart := time.Now()
	generatedCount := 0

	for step := 0; step < opts.MaxTokens; step++ {
		banSpecials(logits, eos)
		if opts.RepetitionPenalty > 1 {
			sampling.ApplyRepetitionPenalty(logits, allTokens, opts.RepetitionPenalty, opts.RepetitionWindow)
		}
		next := sampling.ArgMax(logits)
		if _, stop := eos[next]; stop {
			break
		}
		tok := uint32(next)
		allTokens = append(allTokens, tok)
		generatedCount++
		stream.Push(allTokens, opts.Silent, opts.StreamCallback)
		if opts.NoRepeatNGram > 0 && sampling.HasRepeatedNGram(allTokens[len(ids):], opts.NoRepeatNGram) {
			break
		}

		logits, err = m.ForwardTokens([]uint32{tok})
		if err != nil {
			metrics := buildMetrics(m, len(ids), generatedCount, prefillElapsed, time.Since(decodeStart))
			return stream.String(), metrics, fmt.Errorf("decode step %d: %w", step, err)
		}
	}
	decodeElapsed := time.Since(decodeStart)

	metrics := buildMetrics(m, len(ids), generatedCount, prefillElapsed, decodeElapsed)
	if generatedCount > 0 && opts.PrintMetrics && !opts.Silent {
		fmt.Print(metrics.FormatFooter())
	}
	if !opts.Silent {
		fmt.Println()
	}
	return stream.String(), metrics, nil
}

func (m *Model) generateHybridGPUSample(
	he *fusedgpu.HybridEngine,
	encode func(text string, addSpecial bool) []uint32,
	decode func(ids []uint32, skipSpecial bool) string,
	ids []uint32,
	eos map[int]struct{},
	stream *Streamer,
	allTokens []uint32,
	opts GenOptions,
	prefillStart time.Time,
) (string, GenMetrics, error) {
	_ = encode
	_ = decode
	var zero GenMetrics
	tok, err := he.PrefillSample(ids)
	m.Quiet = false
	prefillElapsed := time.Since(prefillStart)
	if err != nil {
		return "", zero, fmt.Errorf("prefill: %w", err)
	}
	if !opts.Silent {
		fmt.Printf("  prompt loaded in %s (%.2f tok/s) [gpu sample]\nAssistant: ",
			prefillElapsed.Round(time.Millisecond),
			float64(len(ids))/math.Max(prefillElapsed.Seconds(), 1e-9))
	}

	// Chunking didn't help on M5 (compute-bound); keep 1-token sync for lower latency.
	decodeStart := time.Now()
	generatedCount := 0
	for generatedCount < opts.MaxTokens {
		if _, stop := eos[int(tok)]; stop {
			break
		}
		allTokens = append(allTokens, tok)
		generatedCount++
		stream.Push(allTokens, opts.Silent, opts.StreamCallback)
		if opts.NoRepeatNGram > 0 && sampling.HasRepeatedNGram(allTokens[len(ids):], opts.NoRepeatNGram) {
			break
		}

		if generatedCount >= opts.MaxTokens {
			break
		}
		if he.Pos() >= he.MaxSeq() {
			break
		}
		next, err := he.DecodeChunk(1)
		if err != nil {
			metrics := buildMetrics(m, len(ids), generatedCount, prefillElapsed, time.Since(decodeStart))
			return stream.String(), metrics, fmt.Errorf("decode step %d: %w", generatedCount, err)
		}
		tok = next[0]
	}
	decodeElapsed := time.Since(decodeStart)
	metrics := buildMetrics(m, len(ids), generatedCount, prefillElapsed, decodeElapsed)
	if generatedCount > 0 && opts.PrintMetrics && !opts.Silent {
		fmt.Print(metrics.FormatFooter())
	}
	if !opts.Silent {
		fmt.Println()
	}
	return stream.String(), metrics, nil
}

func buildMetrics(m *Model, promptTokens, generatedCount int, prefillElapsed, decodeElapsed time.Duration) GenMetrics {
	metrics := GenMetrics{
		PrefillTime:     prefillElapsed,
		DecodeTime:      decodeElapsed,
		PrefillTokens:   promptTokens,
		GeneratedTokens: generatedCount,
	}
	if generatedCount > 0 {
		if decodeElapsed > 0 {
			metrics.DecodeTokPerSec = float64(generatedCount) / decodeElapsed.Seconds()
		}
		totalElapsed := prefillElapsed + decodeElapsed
		if totalElapsed > 0 {
			metrics.TotalTokPerSec = float64(promptTokens+generatedCount) / totalElapsed.Seconds()
		}
		if promptTokens > 0 && prefillElapsed > 0 {
			metrics.PrefillTokPerSec = float64(promptTokens) / prefillElapsed.Seconds()
		}
	}
	if m != nil {
		fp := m.MemFootprint()
		metrics.HostMB = fp.HostMB
		metrics.VRAMMB = fp.VRAMMB
		metrics.HeapMB = fp.HeapMB
		metrics.WeightsMB = fp.WeightsMB
	}
	return metrics
}

// banSpecials masks ChatML control tokens so greedy decode cannot emit a new
// turn mid-reply. EOS ids stay unmasked so generation can stop cleanly.
func banSpecials(logits []float32, eos map[int]struct{}) {
	// Common ChatML / Qwen specials (ids differ by vocab; only mask in-range).
	// SmolLM2: <|im_start|>=1, <|im_end|>=2. Qwen: im_end=151645, im_start=151644.
	candidates := []int{1, 151644, 151643 /* <|endoftext|> qwen */}
	var ban []int
	for _, id := range candidates {
		if _, isEOS := eos[id]; isEOS {
			continue
		}
		if id >= 0 && id < len(logits) {
			ban = append(ban, id)
		}
	}
	sampling.BanIDs(logits, ban)
}
