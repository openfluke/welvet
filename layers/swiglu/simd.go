package swiglu

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/simd"
)

// ForwardSIMD — Gate/Up/Down via dense SIMD; SiLU⊙ via simd.SiluMulF32.
func ForwardSIMD[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if !simd.Enabled() {
		return nil, nil, fmt.Errorf("swiglu: BackendSIMD but Plan 9 SIMD not enabled")
	}
	return forwardHost(l, input, true)
}

// BackwardSIMD — reverse of ForwardSIMD (simd.SiluMulBwdF32 combine).
func BackwardSIMD[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if !simd.Enabled() {
		return nil, nil, fmt.Errorf("swiglu: BackendSIMD but Plan 9 SIMD not enabled")
	}
	return backwardHost(l, gradOut, input, pre, true)
}
