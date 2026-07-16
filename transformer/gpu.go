package transformer

import (
	"fmt"

	"github.com/openfluke/welvet/fusedgpu"
)

// SyncGPU uploads weights to the full fused WebGPU decoder (Q4 baked only).
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
	return nil
}

// CloseGPU releases the fused GPU engine.
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
