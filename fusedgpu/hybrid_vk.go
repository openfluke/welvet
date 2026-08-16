package fusedgpu

import "fmt"

// HybridVKEngine is the BinaryG128 hybrid decoder on native Vulkan compute (Android).
type HybridVKEngine struct {
	e *hybridVKEngine
}

// Close releases GPU resources for this engine.
func (eng *HybridVKEngine) Close() {
	if eng == nil || eng.e == nil {
		return
	}
	eng.e.release()
	eng.e = nil
}

// Reset clears KV state and position for a new prompt.
func (eng *HybridVKEngine) Reset() error {
	if eng == nil || eng.e == nil {
		return fmt.Errorf("fusedgpu: nil hybrid vk engine")
	}
	return eng.e.resetState()
}

// AppendTokens runs one or more forward steps; returns logits for the last token.
func (eng *HybridVKEngine) AppendTokens(ids []uint32) ([]float32, error) {
	if eng == nil || eng.e == nil {
		return nil, fmt.Errorf("fusedgpu: nil hybrid vk engine")
	}
	return eng.e.appendTokens(ids)
}

// PrefillSample runs the prompt on-device and returns the greedy next token.
func (eng *HybridVKEngine) PrefillSample(ids []uint32) (uint32, error) {
	if eng == nil || eng.e == nil {
		return 0, fmt.Errorf("fusedgpu: nil hybrid vk engine")
	}
	return eng.e.prefillSample(ids)
}

// DecodeSample embeds tok, runs one decode step, returns the next greedy token.
func (eng *HybridVKEngine) DecodeSample(tok uint32) (uint32, error) {
	if eng == nil || eng.e == nil {
		return 0, fmt.Errorf("fusedgpu: nil hybrid vk engine")
	}
	return eng.e.stepTokenSample(tok)
}

// DecodeChunk runs k decode steps (one submit per token on Android).
func (eng *HybridVKEngine) DecodeChunk(k int) ([]uint32, error) {
	if eng == nil || eng.e == nil {
		return nil, fmt.Errorf("fusedgpu: nil hybrid vk engine")
	}
	return eng.e.decodeChunkSample(k)
}

// Pos returns the current sequence position.
func (eng *HybridVKEngine) Pos() int {
	if eng == nil || eng.e == nil {
		return 0
	}
	return eng.e.pos
}

// MaxSeq returns the engine context limit.
func (eng *HybridVKEngine) MaxSeq() int {
	if eng == nil || eng.e == nil {
		return 0
	}
	return eng.e.maxSeq
}

// AdapterName returns the bound GPU device name.
func (eng *HybridVKEngine) AdapterName() string {
	if eng == nil || eng.e == nil {
		return ""
	}
	return eng.e.adapterName
}

// VRAMBytes returns allocated device buffer bytes for this engine.
func (eng *HybridVKEngine) VRAMBytes() uint64 {
	if eng == nil || eng.e == nil {
		return 0
	}
	return eng.e.estimateVRAM()
}
