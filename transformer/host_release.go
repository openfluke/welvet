package transformer

import (
	"fmt"
	"runtime"
	"runtime/debug"

	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
)

// releaseHostPackedWeights drops host packed payloads after a successful GPU fuse
// upload so RAM + Metal buffers aren't double-resident (critical on iPad 12GB for 27B).
// ExportHybridFusedGPUSpec may already have moved payloads; this is the final sweep + flag.
func (m *Model) releaseHostPackedWeights() {
	if m == nil {
		return
	}
	if m.hostWeightsReleased {
		runtime.GC()
		debug.FreeOSMemory()
		fp := m.MemFootprint()
		fmt.Printf("  host weights released after GPU fuse → %s\n", fp.String())
		return
	}
	drop := func(b *quant.Blob) {
		if b == nil {
			return
		}
		b.Raw = nil
		b.Scales = nil
		b.Mins = nil
		b.Q4Packed = nil
		b.F32Cache = nil
	}
	drop(m.embedPacked)
	drop(m.lmHeadPacked)
	if m.lmHead != nil {
		drop(m.lmHead.Packed)
		m.lmHead.Native = nil
	}
	for i := range m.Blocks {
		b := &m.Blocks[i]
		releaseDensePacked(b.Q)
		releaseDensePacked(b.K)
		releaseDensePacked(b.V)
		releaseDensePacked(b.O)
		if b.FFN != nil {
			releaseDensePacked(b.FFN.Gate)
			releaseDensePacked(b.FFN.Up)
			releaseDensePacked(b.FFN.Down)
		}
		if b.Attn != nil {
			releaseDensePacked(b.Attn.Q)
			releaseDensePacked(b.Attn.K)
			releaseDensePacked(b.Attn.V)
			releaseDensePacked(b.Attn.O)
		}
		if b.GDN != nil {
			drop(b.GDN.InQKV)
			drop(b.GDN.InZ)
			drop(b.GDN.InB)
			drop(b.GDN.InA)
			drop(b.GDN.Out)
		}
	}
	m.hostWeightsReleased = true
	runtime.GC()
	debug.FreeOSMemory()
	fp := m.MemFootprint()
	fmt.Printf("  host weights released after GPU fuse → %s\n", fp.String())
}

func releaseDensePacked(l *dense.Layer) {
	if l == nil || l.Weights == nil {
		return
	}
	if b := l.Weights.Packed; b != nil {
		b.Raw = nil
		b.Scales = nil
		b.Mins = nil
		b.Q4Packed = nil
		b.F32Cache = nil
	}
	l.Weights.Native = nil
}

// ensureHostPackedWeights reloads packed blobs from EntityPath if they were released for GPU.
func (m *Model) ensureHostPackedWeights() error {
	if m == nil || !m.hostWeightsReleased {
		return nil
	}
	if m.EntityPath == "" {
		return fmt.Errorf("host weights were released for GPU; remount the .entity to use CPU/SIMD")
	}
	fresh, err := LoadEntity(m.EntityPath)
	if err != nil {
		return fmt.Errorf("reload host weights: %w", err)
	}
	m.stealHostWeightsFrom(fresh)
	m.hostWeightsReleased = false
	fmt.Printf("  host weights reloaded from entity for CPU/SIMD\n")
	return nil
}

func (m *Model) stealHostWeightsFrom(src *Model) {
	if m == nil || src == nil {
		return
	}
	m.Embed = src.Embed
	m.embedPacked = src.embedPacked
	m.FinalNorm = src.FinalNorm
	m.lmHead = src.lmHead
	m.lmHeadPacked = src.lmHeadPacked
	if len(m.Blocks) != len(src.Blocks) {
		m.Blocks = src.Blocks
		return
	}
	for i := range m.Blocks {
		dst, s := &m.Blocks[i], &src.Blocks[i]
		dst.AttnNorm, dst.FFNNorm = s.AttnNorm, s.FFNNorm
		dst.Attn, dst.FFN = s.Attn, s.FFN
		dst.GDN = s.GDN
		dst.Q, dst.K, dst.V, dst.O = s.Q, s.K, s.V, s.O
		dst.QNorm, dst.KNorm = s.QNorm, s.KNorm
		// keep dst KV caches / offset
	}
}
