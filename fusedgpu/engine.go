package fusedgpu

import (
	"fmt"
)

// Engine is a full Q4 decoder on WebGPU (one compute pass per token).
type Engine struct {
	e *engine
}

// Close releases GPU resources.
func (eng *Engine) Close() {
	if eng == nil || eng.e == nil {
		return
	}
	eng.e.release()
	eng.e = nil
}

// Reset clears KV caches and position for a new prompt.
func (eng *Engine) Reset() error {
	if eng == nil || eng.e == nil {
		return fmt.Errorf("fusedgpu: nil engine")
	}
	eng.e.resetState()
	return nil
}

// AppendTokens runs one or more forward steps and returns logits for the last token.
func (eng *Engine) AppendTokens(ids []uint32) ([]float32, error) {
	if eng == nil || eng.e == nil {
		return nil, fmt.Errorf("fusedgpu: nil engine")
	}
	return eng.e.appendTokens(ids)
}

// AdapterName returns the bound GPU adapter (empty if unknown).
func (eng *Engine) AdapterName() string {
	if eng == nil || eng.e == nil || eng.e.adapter == nil {
		return ""
	}
	return eng.e.adapter.GetInfo().Name
}
