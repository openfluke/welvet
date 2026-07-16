package transformer

import (
	"fmt"

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
	spec, err := m.ExportFusedGPUSpec()
	if err != nil {
		return err
	}
	eng, err := fusedgpu.NewFromSpec(spec)
	if err != nil {
		return fmt.Errorf("fusedgpu: %w", err)
	}
	m.gpu = eng
	m.clearFloatCaches() // drop inflate scratch used for Q4 projection
	return nil
}

// clearFloatCaches frees host F32Cache / Int8QS inflate views after GPU upload.
func (m *Model) clearFloatCaches() {
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

// CloseGPU releases the fused GPU engine and encourages VRAM reclaim.
func (m *Model) CloseGPU() {
	if m == nil || m.gpu == nil {
		return
	}
	if eng, ok := m.gpu.(*fusedgpu.Engine); ok {
		eng.Close()
	}
	m.gpu = nil
}

// ForwardTokensGPU runs the fused GPU path when synced; falls back to host ForwardTokens.
func (m *Model) ForwardTokensGPU(ids []uint32) ([]float32, error) {
	if m == nil {
		return nil, fmt.Errorf("transformer: nil model")
	}
	if eng, ok := m.gpu.(*fusedgpu.Engine); ok && eng != nil {
		return eng.AppendTokens(ids)
	}
	return m.ForwardTokens(ids)
}

// GPUAdapterName returns the fused GPU adapter name when active.
func (m *Model) GPUAdapterName() string {
	if eng, ok := m.gpu.(*fusedgpu.Engine); ok && eng != nil {
		return eng.AdapterName()
	}
	return ""
}

// ResetKV clears attention caches; also resets fused GPU state when active.
func (m *Model) ResetKV() {
	if m == nil {
		return
	}
	for i := range m.Blocks {
		m.Blocks[i].Attn.KVOffset = 0
		m.Blocks[i].Attn.KVCacheK = nil
		m.Blocks[i].Attn.KVCacheV = nil
		m.Blocks[i].Attn.DecodeScratchQ = nil
		m.Blocks[i].Attn.DecodeScratchAttn = nil
	}
	if eng, ok := m.gpu.(*fusedgpu.Engine); ok && eng != nil {
		_ = eng.Reset()
	}
}
