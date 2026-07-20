package transformer

import (
	"context"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/openfluke/welvet/fusedgpu"
	"github.com/openfluke/welvet/model/sampling"
)

const thinkTokenBudget = 256 // max tokens inside <think> before we force-close + answer

// GenOptions controls generation + streaming.
type GenOptions struct {
	MaxTokens int
	// Context cancels generation between model steps. A nil context is treated
	// as context.Background().
	Context context.Context
	// Silent suppresses stdout streaming (metrics footer still prints unless PrintMetrics is false).
	Silent bool
	// PrintMetrics prints the Lucy-style tok/s footer when generation produced tokens.
	PrintMetrics bool
	// StreamCallback receives each streamed text chunk (after ChatML cleanup).
	StreamCallback func(piece string)
	// Temperature scales logits before TopK (≤0 → greedy ArgMax). Default 0 (greedy).
	Temperature float32
	// TopK keeps the K highest logits after temperature (≤0 = full vocab; 1 = greedy).
	TopK int
	// Deterministic forces ArgMax even when Temperature/TopK would sample.
	Deterministic bool
	// BannedTokens are never sampled (host path; hybrid GPU sample skips host masks).
	BannedTokens []int
	// RepetitionPenalty > 1 down-weights recent tokens (0 = default 1.15; <0 disables).
	RepetitionPenalty float32
	// RepetitionWindow is how many recent tokens the penalty considers (0 = 64).
	RepetitionWindow int
	// NoRepeatNGram stops when the last n tokens repeat immediately (0 = 8; <0 disables).
	NoRepeatNGram int
	// EnableThinking lets Qwen3/Bonsai emit <think>…</think> (default false = hard-disable).
	// Ignored on non-thinking models (SmolLM, etc.).
	EnableThinking bool
}

// Generate runs decode for one user message (ChatML by default).
// Default sampling is greedy (Temperature≤0 / TopK=1 / Deterministic).
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
	if opts.Context == nil {
		opts.Context = context.Background()
	}
	if err := opts.Context.Err(); err != nil {
		return "", zero, err
	}
	if !opts.Silent && !opts.PrintMetrics {
		opts.PrintMetrics = true
	}
	if opts.RepetitionPenalty == 0 {
		opts.RepetitionPenalty = 1.15
	}
	if opts.EnableThinking && m.SupportsThinking() && opts.RepetitionPenalty < 1.2 {
		opts.RepetitionPenalty = 1.2
	}
	if opts.RepetitionWindow <= 0 {
		opts.RepetitionWindow = 64
	}
	if opts.NoRepeatNGram == 0 {
		opts.NoRepeatNGram = 8
	}

	prompt := ChatML.BuildPrompt(turns, systemPrompt, userMsg)
	hideThink := false
	if m.SupportsThinking() {
		if opts.EnableThinking {
			// Soft /think only — no hard-open <think> (that traps 4B/8B).
			prompt = ChatML.BuildPromptThink(turns, systemPrompt, userMsg)
		} else {
			prompt = ChatML.BuildPromptNoThink(turns, systemPrompt, userMsg)
			hideThink = true
		}
	}
	ids := encode(prompt, false)
	if len(ids) == 0 {
		return "", zero, fmt.Errorf("tokenizer produced empty prompt")
	}

	streamDecode := func(toks []uint32) string { return decode(toks, false) }
	allTokens := append([]uint32(nil), ids...)
	stream := NewStreamer(streamDecode, ids)
	stream.HideThink = hideThink

	eos := make(map[int]struct{}, len(m.EOSTokens))
	for _, t := range m.EOSTokens {
		eos[t] = struct{}{}
	}

	m.ResetKV()
	m.Quiet = opts.Silent
	prefillStart := time.Now()

	if he, ok := m.gpu.(*fusedgpu.HybridEngine); ok && he != nil {
		return m.generateHybridGPUSample(he, encode, decode, ids, eos, stream, allTokens, opts, prefillStart)
	}

	logits, err := m.ForwardTokens(ids)
	m.Quiet = false
	prefillElapsed := time.Since(prefillStart)
	if err != nil {
		return "", zero, fmt.Errorf("prefill: %w", err)
	}
	if err := opts.Context.Err(); err != nil {
		return "", buildMetrics(m, len(ids), 0, prefillElapsed, 0), err
	}
	if !opts.Silent {
		fmt.Printf("  prompt loaded in %s (%.2f tok/s)\nAssistant: ",
			prefillElapsed.Round(time.Millisecond),
			float64(len(ids))/math.Max(prefillElapsed.Seconds(), 1e-9))
	}

	decodeStart := time.Now()
	generatedCount := 0
	thinkClosed := false

	for generatedCount < opts.MaxTokens {
		if err := opts.Context.Err(); err != nil {
			metrics := buildMetrics(m, len(ids), generatedCount, prefillElapsed, time.Since(decodeStart))
			return finalizeAssistantReply(stream.String(), opts.EnableThinking && m.SupportsThinking()), metrics, err
		}
		banSpecials(logits, eos)
		if len(opts.BannedTokens) > 0 {
			sampling.BanIDs(logits, opts.BannedTokens)
		}
		if opts.RepetitionPenalty > 1 {
			sampling.ApplyRepetitionPenalty(logits, allTokens, opts.RepetitionPenalty, opts.RepetitionWindow)
		}
		next := sampling.SampleTopK(logits, opts.TopK, opts.Temperature, opts.Deterministic)
		if _, stop := eos[next]; stop {
			break
		}
		tok := uint32(next)
		allTokens = append(allTokens, tok)
		generatedCount++
		stream.Push(allTokens, opts.Silent, opts.StreamCallback)
		gen := allTokens[len(ids):]

		if opts.EnableThinking && stream.InThink() && !thinkClosed &&
			(shouldStopRepeat(gen, opts) || thinkLen(stream) >= thinkTokenBudget) {
			closeIDs := encode("</think>\n\n", false)
			stream.ForceCloseThink(opts.Silent, opts.StreamCallback)
			thinkClosed = true
			if len(closeIDs) > 0 {
				logits, err = m.ForwardTokens(closeIDs)
				if err != nil {
					metrics := buildMetrics(m, len(ids), generatedCount, prefillElapsed, time.Since(decodeStart))
					return finalizeAssistantReply(stream.String(), true), metrics, fmt.Errorf("close think: %w", err)
				}
				allTokens = append(allTokens, closeIDs...)
			}
			continue
		}
		if !stream.InThink() && shouldStopRepeat(gen, opts) {
			break
		}

		logits, err = m.ForwardTokens([]uint32{tok})
		if err != nil {
			metrics := buildMetrics(m, len(ids), generatedCount, prefillElapsed, time.Since(decodeStart))
			return stream.String(), metrics, fmt.Errorf("decode step %d: %w", generatedCount, err)
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
	return finalizeAssistantReply(stream.String(), opts.EnableThinking && m.SupportsThinking()), metrics, nil
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
	var zero GenMetrics
	tok, err := he.PrefillSample(ids)
	m.Quiet = false
	prefillElapsed := time.Since(prefillStart)
	if err != nil {
		return "", zero, fmt.Errorf("prefill: %w", err)
	}
	if err := opts.Context.Err(); err != nil {
		return "", buildMetrics(m, len(ids), 0, prefillElapsed, 0), err
	}
	if !opts.Silent {
		fmt.Printf("  prompt loaded in %s (%.2f tok/s) [gpu sample]\nAssistant: ",
			prefillElapsed.Round(time.Millisecond),
			float64(len(ids))/math.Max(prefillElapsed.Seconds(), 1e-9))
	}

	decodeStart := time.Now()
	generatedCount := 0
	thinkClosed := false

	for generatedCount < opts.MaxTokens {
		if err := opts.Context.Err(); err != nil {
			metrics := buildMetrics(m, len(ids), generatedCount, prefillElapsed, time.Since(decodeStart))
			return finalizeAssistantReply(stream.String(), opts.EnableThinking), metrics, err
		}
		if _, stop := eos[int(tok)]; stop {
			break
		}
		allTokens = append(allTokens, tok)
		generatedCount++
		stream.Push(allTokens, opts.Silent, opts.StreamCallback)
		gen := allTokens[len(ids):]

		if opts.EnableThinking && stream.InThink() && !thinkClosed &&
			(shouldStopRepeat(gen, opts) || thinkLen(stream) >= thinkTokenBudget) {
			// Force-close think, re-prefill so far + </think>, then continue for the answer.
			closeIDs := encode("</think>\n\n", false)
			stream.ForceCloseThink(opts.Silent, opts.StreamCallback)
			thinkClosed = true
			allTokens = append(allTokens, closeIDs...)
			if err := he.Reset(); err != nil {
				metrics := buildMetrics(m, len(ids), generatedCount, prefillElapsed, time.Since(decodeStart))
				return finalizeAssistantReply(stream.String(), true), metrics, err
			}
			tok, err = he.PrefillSample(allTokens)
			if err != nil {
				metrics := buildMetrics(m, len(ids), generatedCount, prefillElapsed, time.Since(decodeStart))
				return finalizeAssistantReply(stream.String(), true), metrics, fmt.Errorf("reprefill after think: %w", err)
			}
			continue
		}
		if !stream.InThink() && shouldStopRepeat(gen, opts) {
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
	return finalizeAssistantReply(stream.String(), opts.EnableThinking), metrics, nil
}

func thinkLen(s *Streamer) int {
	if s == nil {
		return 0
	}
	t := s.String()
	start := strings.Index(t, "<think>")
	if start < 0 {
		return 0
	}
	body := t[start+len("<think>"):]
	if end := strings.Index(body, "</think>"); end >= 0 {
		body = body[:end]
	}
	// Rough token proxy: ~4 chars/token.
	return len(body) / 4
}

// shouldStopRepeat stops classic adjacent n-gram loops and recent lookback loops.
func shouldStopRepeat(gen []uint32, opts GenOptions) bool {
	n := opts.NoRepeatNGram
	if n <= 0 {
		return false
	}
	if sampling.HasRepeatedNGram(gen, n) {
		return true
	}
	look := 256
	if opts.EnableThinking {
		look = 320
	}
	if n >= 6 && sampling.HasRepeatedNGramRecent(gen, n, look) {
		return true
	}
	if opts.EnableThinking && sampling.HasRepeatedNGramRecent(gen, 10, look) {
		return true
	}
	return false
}

func finalizeAssistantReply(reply string, thinking bool) string {
	reply = strings.TrimSpace(reply)
	if !thinking || reply == "" {
		return reply
	}
	if strings.Contains(reply, "<think>") && !strings.Contains(reply, "</think>") {
		return reply + "\n</think>"
	}
	return reply
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

func banSpecials(logits []float32, eos map[int]struct{}) {
	candidates := []int{1, 151644, 151643}
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
