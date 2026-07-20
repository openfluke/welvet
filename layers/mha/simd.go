package mha

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/simd"
)

// ForwardSIMD — Q/K/V/O via dense SIMD (syncProjExec); RoPE + causal attn on host.
func ForwardSIMD[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if !simd.Enabled() {
		return nil, nil, fmt.Errorf("mha: BackendSIMD but Plan 9 SIMD not enabled")
	}
	return forwardHost(l, input)
}

// BackwardSIMD — reverse of ForwardSIMD.
func BackwardSIMD[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if !simd.Enabled() {
		return nil, nil, fmt.Errorf("mha: BackendSIMD but Plan 9 SIMD not enabled")
	}
	return backwardHost(l, gradOut, input, pre)
}
