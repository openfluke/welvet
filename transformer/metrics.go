package transformer

import (
	"fmt"
	"time"
)

// GenMetrics holds generation timing (Lucy / poly parity).
type GenMetrics struct {
	PrefillTime      time.Duration
	DecodeTime       time.Duration
	PrefillTokens    int
	GeneratedTokens  int
	PrefillTokPerSec float64
	DecodeTokPerSec  float64
	TotalTokPerSec   float64
}

// FormatFooter returns the Lucy-style one-line speed summary.
func (m GenMetrics) FormatFooter() string {
	if m.GeneratedTokens <= 0 {
		return ""
	}
	return fmt.Sprintf(
		"\n\n(prefill: %.2f tok/s, %d prompt tokens | decode: %.2f tok/s, %d generated | total: %.2f tok/s)\n",
		m.PrefillTokPerSec,
		m.PrefillTokens,
		m.DecodeTokPerSec,
		m.GeneratedTokens,
		m.TotalTokPerSec,
	)
}
