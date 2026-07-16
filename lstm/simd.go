package lstm

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/simd"
)

// ForwardSIMD — Dense SIMD MatVec per gate/timestep.
func ForwardSIMD[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if !simd.Enabled() {
		return nil, nil, fmt.Errorf("lstm: BackendSIMD but Plan 9 SIMD not enabled")
	}
	return forwardViaDense(l, input)
}

// BackwardSIMD — BPTT via Dense SIMD.
func BackwardSIMD[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if !simd.Enabled() {
		return nil, nil, fmt.Errorf("lstm: BackendSIMD but Plan 9 SIMD not enabled")
	}
	return backwardViaDense(l, gradOut, input, pre)
}
