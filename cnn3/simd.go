package cnn3

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/simd"
)

// ForwardSIMD — im2col + Dense SIMD path.
func ForwardSIMD[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if !simd.Enabled() {
		return nil, nil, fmt.Errorf("cnn3: BackendSIMD but Plan 9 SIMD not enabled")
	}
	return forwardViaDense(l, input)
}

// BackwardSIMD — reverse of ForwardSIMD.
func BackwardSIMD[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if !simd.Enabled() {
		return nil, nil, fmt.Errorf("cnn3: BackendSIMD but Plan 9 SIMD not enabled")
	}
	return backwardViaDense(l, gradOut, input, pre)
}
