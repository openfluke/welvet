package mha

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/webgpu"
)

// ForwardWebGPU — real device required (no host fake). Q/K/V/O projections
// dispatch to dense.ForwardWebGPU (on-device GEMV, since syncProjExec already
// propagated Exec.Backend = BackendWebGPU to the four projections); RoPE,
// QK-norm and the causal softmax-attention ALU itself stay on host — there is
// no on-device attention kernel yet. Suite note: "attn host; proj DenseGEMV on-device".
func ForwardWebGPU[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("mha: BackendWebGPU but no device (no host fake)")
	}
	return forwardHost(l, input)
}

// BackwardWebGPU — reverse of ForwardWebGPU; same on-device/host split (proj
// GEMV/GEMVT on device, attention backward ALU on host).
func BackwardWebGPU[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("mha: BackendWebGPU but no device (no host fake)")
	}
	return backwardHost(l, gradOut, input, pre)
}
