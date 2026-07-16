package mha

import "github.com/openfluke/welvet/core"

// ForwardSIMD — Q/K/V/O via dense SIMD; RoPE + causal attn on host (activation ALU).
func ForwardSIMD[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	return forwardHost(l, input)
}

// BackwardSIMD — reverse of ForwardSIMD.
func BackwardSIMD[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	return backwardHost(l, gradOut, input, pre)
}
