package gdn

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/webgpu"
)

// ForwardWebGPU requires a real device (gate), then runs the host decode loop —
// GDN's ForwardDecode already optionally stages projections through WebGPU GEMV
// per-token when l.UseGPU is set (see layer.go matVec). No silent CPU fallback
// when BackendWebGPU is requested but no device is present.
func ForwardWebGPU[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("gdn: BackendWebGPU but no device (no host fake)")
	}
	return Forward(l, input)
}

// BackwardWebGPU requires a real device (gate), then runs the host truncated-BPTT
// backward (see train.go).
func BackwardWebGPU[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("gdn: BackendWebGPU but no device (no host fake)")
	}
	return Backward(l, gradOut, input, pre)
}
