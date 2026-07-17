package transformer

import (
	"fmt"
	"runtime"

	"github.com/openfluke/welvet/dense"
	"github.com/openfluke/welvet/fusedgpu"
	"github.com/openfluke/welvet/quant"
)

// MemFootprint is a point-in-time host + GPU memory snapshot for chat UIs.
type MemFootprint struct {
	HostMB    float64 // process RSS when available, else Go heap Sys
	HeapMB    float64 // runtime.MemStats.Sys
	VRAMMB    float64 // fused WebGPU buffer bytes (0 if CPU/SIMD only)
	WeightsMB float64 // packed weight blobs still resident on host
	Backend   string
}

func (f MemFootprint) String() string {
	// HostMB is process RSS (Go heap + Vulkan driver maps + tokenizer + …),
	// not “weights still on RAM”. WeightsMB is the packed-blob residual.
	if f.VRAMMB > 0 {
		return fmt.Sprintf("%.0f MB RSS (%.0f MB host weights) · %.0f MB GPU", f.HostMB, f.WeightsMB, f.VRAMMB)
	}
	if f.WeightsMB > 0 {
		return fmt.Sprintf("%.0f MB RSS (%.0f MB host weights)", f.HostMB, f.WeightsMB)
	}
	return fmt.Sprintf("%.0f MB RSS", f.HostMB)
}

// MemFootprint reports host RAM + estimated GPU buffer usage for the loaded model.
func (m *Model) MemFootprint() MemFootprint {
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	heapMB := float64(ms.Sys) / (1024 * 1024)
	hostMB := heapMB
	if rss := processRSSBytes(); rss > 0 {
		hostMB = float64(rss) / (1024 * 1024)
	}
	fp := MemFootprint{
		HostMB:    hostMB,
		HeapMB:    heapMB,
		WeightsMB: float64(m.hostPackedBytes()) / (1024 * 1024),
		Backend:   fmt.Sprintf("%v", m.Exec.Backend),
	}
	fp.VRAMMB = float64(m.gpuBufferBytes()) / (1024 * 1024)
	return fp
}

func (m *Model) gpuBufferBytes() uint64 {
	if m == nil || m.gpu == nil {
		return 0
	}
	switch g := m.gpu.(type) {
	case *fusedgpu.HybridEngine:
		return g.VRAMBytes()
	case *fusedgpu.Engine:
		return g.VRAMBytes()
	default:
		return 0
	}
}

func (m *Model) hostPackedBytes() uint64 {
	if m == nil {
		return 0
	}
	var n uint64
	addBlob := func(b *quant.Blob) {
		if b == nil {
			return
		}
		n += uint64(len(b.Raw))
		n += uint64(len(b.Scales) * 4)
		n += uint64(len(b.F32Cache) * 4)
	}
	addDense := func(l *dense.Layer) {
		if l == nil || l.Weights == nil {
			return
		}
		addBlob(l.Weights.Packed)
	}
	addBlob(m.embedPacked)
	addBlob(m.lmHeadPacked)
	for i := range m.Blocks {
		b := &m.Blocks[i]
		if b.FFN != nil {
			addDense(b.FFN.Gate)
			addDense(b.FFN.Up)
			addDense(b.FFN.Down)
		}
		addDense(b.Q)
		addDense(b.K)
		addDense(b.V)
		addDense(b.O)
		if b.GDN != nil {
			addBlob(b.GDN.InQKV)
			addBlob(b.GDN.InZ)
			addBlob(b.GDN.InB)
			addBlob(b.GDN.InA)
			addBlob(b.GDN.Out)
		}
	}
	return n
}
