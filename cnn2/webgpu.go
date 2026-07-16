package cnn2

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/webgpu"
)

// ForwardWebGPU — requires a real device; im2col host + Dense WebGPU GEMV.
func ForwardWebGPU[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("cnn2: BackendWebGPU but no device (no host fake)")
	}
	return forwardViaDense(l, input)
}

// BackwardWebGPU — reverse of ForwardWebGPU.
func BackwardWebGPU[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("cnn2: BackendWebGPU but no device (no host fake)")
	}
	return backwardViaDense(l, gradOut, input, pre)
}
