package transformer

import (
	"fmt"
	"runtime"
	"runtime/debug"

	"github.com/openfluke/welvet/dense"
	"github.com/openfluke/welvet/fusedgpu"
	"github.com/openfluke/welvet/quant"
)

// SyncGPU uploads weights to the full fused WebGPU decoder.
// Non-Q4 packs are projected to Q4_0 for upload; host entity format is unchanged.
func (m *Model) SyncGPU() error {
	if m == nil {
		return fmt.Errorf("transformer: nil model")
	}
	if m.gpu != nil {
		return nil
	}
	// Drop simd_fuse inflate views before building the upload bundle — otherwise
	// host RSS stays ~3–4GB and CreateBufferInit staging fails mid-upload (kc_*).
	m.clearFloatCaches()
	runtime.GC()
	debug.FreeOSMemory()

	spec, err := m.ExportFusedGPUSpec()
	if err != nil {
		return err
	}
	m.clearFloatCaches()
	runtime.GC()

	eng, err := fusedgpu.NewFromSpec(spec)
	if err != nil {
		return fmt.Errorf("fusedgpu: %w", err)
	}
	m.gpu = eng
	m.clearFloatCaches()
	return nil
}

// ClearFloatCaches drops host inflate views (F32Cache) to free RAM between profiles.
func (m *Model) ClearFloatCaches() {
	m.clearFloatCaches()
}

func (m *Model) clearFloatCaches() {
	if m == nil {
		return
	}
	clearBlob := func(b *quant.Blob) {
		if b != nil {
			b.F32Cache = nil
		}
	}
	if m.lmHeadPacked != nil {
		clearBlob(m.lmHeadPacked)
	}
	if m.lmHead != nil {
		clearBlob(m.lmHead.Packed)
	}
	for i := range m.Blocks {
		b := &m.Blocks[i]
		for _, d := range []*dense.Layer{b.Attn.Q, b.Attn.K, b.Attn.V, b.Attn.O, b.FFN.Gate, b.FFN.Up, b.FFN.Down} {
			if d != nil && d.Weights != nil {
				clearBlob(d.Weights.Packed)
			}
		}
	}
}

// SyncHybridFused uploads the full BinaryG128 hybrid decoder to GPU (all weights on device).
// Needs ~entity size + scratch VRAM (Bonsai 27B ≈ 5–8+ GiB). No host GEMV fallback.
func (m *Model) SyncHybridFused() error {
	if m == nil {
		return fmt.Errorf("transformer: nil model")
	}
	if m.gpu != nil {
		if _, ok := m.gpu.(*fusedgpu.HybridEngine); ok {
			return nil
		}
		m.CloseGPU()
	}
	m.CloseHybridGPU()
	runtime.GC()
	debug.FreeOSMemory()

	spec, err := m.ExportHybridFusedGPUSpec()
	if err != nil {
		return err
	}
	runtime.GC()

	eng, err := fusedgpu.NewHybridFromSpec(spec)
	if err != nil {
		return fmt.Errorf("fusedgpu hybrid: %w", err)
	}
	m.gpu = eng
	return nil
}

// CloseGPU releases the fused GPU engine and encourages VRAM reclaim.
func (m *Model) CloseGPU() {
	if m == nil || m.gpu == nil {
		return
	}
	switch eng := m.gpu.(type) {
	case *fusedgpu.Engine:
		eng.Close()
	case *fusedgpu.HybridEngine:
		eng.Close()
	}
	m.gpu = nil
}

// ForwardTokensGPU runs the fused GPU path when synced; falls back to host ForwardTokens.
func (m *Model) ForwardTokensGPU(ids []uint32) ([]float32, error) {
	if m == nil {
		return nil, fmt.Errorf("transformer: nil model")
	}
	switch eng := m.gpu.(type) {
	case *fusedgpu.Engine:
		if eng != nil {
			return eng.AppendTokens(ids)
		}
	case *fusedgpu.HybridEngine:
		if eng != nil {
			return eng.AppendTokens(ids)
		}
	}
	return m.forwardTokensHost(ids)
}

// GPUAdapterName returns the fused GPU adapter name when active.
func (m *Model) GPUAdapterName() string {
	switch eng := m.gpu.(type) {
	case *fusedgpu.Engine:
		if eng != nil {
			return eng.AdapterName()
		}
	case *fusedgpu.HybridEngine:
		if eng != nil {
			return eng.AdapterName()
		}
	}
	return ""
}

// ResetKV clears attention caches; also resets fused GPU state when active.
func (m *Model) ResetKV() {
	if m == nil {
		return
	}
	for i := range m.Blocks {
		b := &m.Blocks[i]
		if b.Attn != nil {
			b.Attn.KVOffset = 0
			b.Attn.KVCacheK = nil
			b.Attn.KVCacheV = nil
			b.Attn.DecodeScratchQ = nil
			b.Attn.DecodeScratchAttn = nil
		}
		b.KVOffset = 0
		b.KVCacheK = nil
		b.KVCacheV = nil
		if b.GDN != nil {
			b.GDN.Reset()
		}
	}
	switch eng := m.gpu.(type) {
	case *fusedgpu.Engine:
		if eng != nil {
			_ = eng.Reset()
		}
	case *fusedgpu.HybridEngine:
		if eng != nil {
			_ = eng.Reset()
		}
	}
}
