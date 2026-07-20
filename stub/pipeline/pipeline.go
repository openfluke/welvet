package pipeline

import "fmt"

// PipelineForwardStats summarizes one pipeline (wavefront) forward pass.
type PipelineForwardStats struct {
	PipelineTicks     uint64
	SubLayerOps       int
	MaxActiveJobs     int
	MaxBlockSpread    int
	MaxDistinctBlocks int
	MaxPendingTokens  int
	StallFallback     bool
	TokenDoneTick     []int
}

// TokenTimelineSummary is a readable view of per-token completion skew.
type TokenTimelineSummary struct {
	NumTokens       int
	FirstDoneTick   int
	LastDoneTick    int
	TickSpread      int
	SampleIndices   []int
	SampleDoneTicks []int
}

// SummarizeTokenTimeline builds a summary from pipeline stats.
func (p PipelineForwardStats) SummarizeTokenTimeline() TokenTimelineSummary {
	out := TokenTimelineSummary{}
	if len(p.TokenDoneTick) == 0 {
		return out
	}
	out.NumTokens = len(p.TokenDoneTick)
	out.FirstDoneTick = p.TokenDoneTick[0]
	out.LastDoneTick = p.TokenDoneTick[len(p.TokenDoneTick)-1]
	if out.FirstDoneTick >= 0 && out.LastDoneTick >= 0 {
		out.TickSpread = out.LastDoneTick - out.FirstDoneTick
	}
	for _, q := range []float64{0, 0.25, 0.5, 0.75, 1.0} {
		idx := int(float64(out.NumTokens-1) * q)
		if idx < 0 {
			idx = 0
		}
		if idx >= out.NumTokens {
			idx = out.NumTokens - 1
		}
		out.SampleIndices = append(out.SampleIndices, idx)
		tick := -1
		if idx < len(p.TokenDoneTick) {
			tick = p.TokenDoneTick[idx]
		}
		out.SampleDoneTicks = append(out.SampleDoneTicks, tick)
	}
	return out
}

// FormatComparison renders a human-readable normal vs pipeline comparison.
func (s TokenTimelineSummary) FormatComparison(normalWallSec float64, pipelineTicks uint64) string {
	if s.NumTokens == 0 {
		return "  (no per-token timeline recorded)\n"
	}
	var b string
	b += fmt.Sprintf("  Normal: all %d prompt tokens move through each block together.\n", s.NumTokens)
	b += fmt.Sprintf("          They effectively finish the stack at the same time (~%.2fs wall).\n", normalWallSec)
	b += "  Pipeline: each token is its own job; finishes the stack at different ticks.\n"
	if s.FirstDoneTick >= 0 && s.LastDoneTick >= 0 {
		b += fmt.Sprintf("          token[0] done @ tick %d  |  token[%d] done @ tick %d\n",
			s.FirstDoneTick, s.NumTokens-1, s.LastDoneTick)
		b += fmt.Sprintf("          stagger: %d ticks between first and last token finishing\n", s.TickSpread)
	}
	b += "          sample (token index → done tick):\n            "
	for i, idx := range s.SampleIndices {
		if i > 0 {
			b += "  "
		}
		tick := -1
		if i < len(s.SampleDoneTicks) {
			tick = s.SampleDoneTicks[i]
		}
		b += fmt.Sprintf("[%d]→%d", idx, tick)
	}
	b += "\n"
	if pipelineTicks > 0 && s.TickSpread > 0 {
		frac := float64(s.TickSpread) / float64(pipelineTicks)
		b += fmt.Sprintf("          first→last token spread is %.0f%% of total prefill pipeline ticks\n", frac*100)
	}
	return b
}
