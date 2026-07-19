package convt1

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/webgpu"
)

// ForwardWebGPU — host scatter when device present (no silent host fake).
func ForwardWebGPU[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("convt1: BackendWebGPU but no device (no host fake)")
	}
	return forwardHost(l, input)
}

// BackwardWebGPU — reverse of ForwardWebGPU.
func BackwardWebGPU[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("convt1: BackendWebGPU but no device (no host fake)")
	}
	return backwardHost(l, gradOut, input, pre)
}
