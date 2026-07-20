package memory

import (
	"fmt"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/weights"
)

// Footprint reports accounted host model weights and GPU weights + KV.
type Footprint struct {
	HostWeightsMB float64
	GPUWeightsMB  float64
	GPUKVMB       float64
}

// FromBytes builds a footprint from raw byte counts.
func FromBytes(host, gpuW, gpuKV int64) Footprint {
	mb := float64(1024 * 1024)
	return Footprint{
		HostWeightsMB: float64(host) / mb,
		GPUWeightsMB:  float64(gpuW) / mb,
		GPUKVMB:       float64(gpuKV) / mb,
	}
}

// FromGrid sums Dense weight store sizes (host); GPU fields stay 0 until wired.
func FromGrid(g *architecture.Grid) Footprint {
	if g == nil {
		return Footprint{}
	}
	var host int64
	for _, c := range g.HopOrder() {
		cell := g.At(c.Z, c.Y, c.X, c.L)
		if cell == nil {
			continue
		}
		if op, ok := cell.Op.(*dense.Layer); ok && op.Weights != nil {
			host += storeBytes(op.Weights)
		}
	}
	return FromBytes(host, 0, 0)
}

func storeBytes(s *weights.Store) int64 {
	if s == nil {
		return 0
	}
	return int64(s.Rows * s.Cols * 4) // logical f32 size
}

// FormatOneLine returns a compact summary.
func (m Footprint) FormatOneLine() string {
	return fmt.Sprintf(
		"host weights %.2f MB | GPU weights %.2f MB | GPU KV %.2f MB",
		m.HostWeightsMB, m.GPUWeightsMB, m.GPUKVMB,
	)
}

// FormatDetailed returns a multi-line block.
func (m Footprint) FormatDetailed() string {
	return fmt.Sprintf(
		"Memory: host weights %8.2f MB | GPU weights %8.2f MB | GPU KV %8.2f MB\n",
		m.HostWeightsMB, m.GPUWeightsMB, m.GPUKVMB,
	)
}

// MemoryFootprint is the loom type alias.
type MemoryFootprint = Footprint
