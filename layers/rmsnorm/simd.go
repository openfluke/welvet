package rmsnorm

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/simd"
)

// ForwardSIMD — Σx² via DotTile; affine scale via simd.RMSNormScaleF32.
func ForwardSIMD[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if !simd.Enabled() {
		return nil, nil, fmt.Errorf("rmsnorm: BackendSIMD but Plan 9 SIMD not enabled")
	}
	return forwardHost(l, input, true)
}

// BackwardSIMD — reverse of ForwardSIMD (DotTile stats + simd scale for dx).
func BackwardSIMD[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if !simd.Enabled() {
		return nil, nil, fmt.Errorf("rmsnorm: BackendSIMD but Plan 9 SIMD not enabled")
	}
	return backwardHost(l, gradOut, input, pre, true)
}
