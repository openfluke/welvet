package transformer

import (
	"fmt"
	"time"
)

// GenMetrics holds generation timing (Lucy / poly parity) + memory snapshot.
type GenMetrics struct {
	PrefillTime      time.Duration
	DecodeTime       time.Duration
	PrefillTokens    int
	GeneratedTokens  int
	PrefillTokPerSec float64
	DecodeTokPerSec  float64
	TotalTokPerSec   float64
	HostMB           float64 // process RSS (not host weight blobs)
	VRAMMB           float64 // GPU buffer bytes (0 if CPU/SIMD)
	HeapMB           float64
	WeightsMB        float64 // packed weight blobs still on host (0 after gpu_fuse release)
}

// FormatFooter returns the Lucy-style one-line speed + memory summary.
func (m GenMetrics) FormatFooter() string {
	if m.GeneratedTokens <= 0 {
		return ""
	}
	mem := ""
	if m.HostMB > 0 || m.VRAMMB > 0 {
		if m.VRAMMB > 0 {
			mem = fmt.Sprintf(" | mem: %.0f MB RSS (%.0f MB host weights), %.0f MB GPU", m.HostMB, m.WeightsMB, m.VRAMMB)
		} else {
			mem = fmt.Sprintf(" | mem: %.0f MB RSS (%.0f MB host weights)", m.HostMB, m.WeightsMB)
		}
	}
	return fmt.Sprintf(
		"\n\n(prefill: %.2f tok/s, %d prompt tokens | decode: %.2f tok/s, %d generated | total: %.2f tok/s%s)\n",
		m.PrefillTokPerSec,
		m.PrefillTokens,
		m.DecodeTokPerSec,
		m.GeneratedTokens,
		m.TotalTokPerSec,
		mem,
	)
}
