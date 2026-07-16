package mha

import "github.com/openfluke/welvet/core"

// ForwardWebGPU — Q/K/V/O via dense WebGPU; RoPE + causal attn on host (activation ALU).
func ForwardWebGPU[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	return forwardHost(l, input)
}

// BackwardWebGPU — reverse of ForwardWebGPU.
func BackwardWebGPU[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	return backwardHost(l, gradOut, input, pre)
}
