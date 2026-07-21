package gdn

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/webgpu"
)

// ForwardWebGPU requires a real device (gate), then runs the host decode loop.
// ForwardDecode stages projections through WebGPU GEMV when UseGPU is set (syncExec).
// No silent CPU fallback when BackendWebGPU is requested but no device is present.
func ForwardWebGPU[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("gdn: BackendWebGPU but no device (no host fake)")
	}
	return forwardHost(l, input)
}

// BackwardWebGPU requires a real device (gate), then runs truncated-BPTT host backward.
func BackwardWebGPU[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("gdn: BackendWebGPU but no device (no host fake)")
	}
	return backwardHost(l, gradOut, input, pre)
}
