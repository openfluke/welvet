package rmsnorm

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/webgpu"
)

// ForwardWebGPU — requires a real device (no host fake); RMS ALU on host today.
func ForwardWebGPU[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("rmsnorm: BackendWebGPU but no device (no host fake)")
	}
	return forwardHost(l, input, false)
}

// BackwardWebGPU — reverse of ForwardWebGPU.
func BackwardWebGPU[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("rmsnorm: BackendWebGPU but no device (no host fake)")
	}
	return backwardHost(l, gradOut, input, pre, false)
}
