package swiglu

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/webgpu"
)

// ForwardWebGPU — real device only; Gate/Up/Down projections run via
// dense.ForwardWebGPU, and the SiLU(gate)⊙up fuse itself runs on-device too
// (webgpu.SwiGLUFuse) — no host math in the forward path.
func ForwardWebGPU[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	return forwardWebGPUCompose(l, input)
}

func forwardWebGPUCompose[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("swiglu: BackendWebGPU but no device (no host fake)")
	}
	l.syncProjExec()
	lay, err := parseLayout(l.Cfg.InputDim, input)
	if err != nil {
		return nil, nil, err
	}
	flat := flatten(input, lay)
	inter := l.Cfg.IntermediateDim
	bs := lay.batch * lay.seq

	gatePre, _, err := dense.ForwardWebGPU(l.Gate, flat)
	if err != nil {
		return nil, nil, fmt.Errorf("swiglu Gate: %w", err)
	}
	_, upPost, err := dense.ForwardWebGPU(l.Up, flat)
	if err != nil {
		return nil, nil, fmt.Errorf("swiglu Up: %w", err)
	}

	gateF := core.SliceAsFloat32(gatePre.Data)
	upF := core.SliceAsFloat32(upPost.Data)
	hF := make([]float32, bs*inter)
	if err := webgpu.SwiGLUFuse(gateF, upF, hF, bs*inter); err != nil {
		return nil, nil, err
	}
	h := core.NewTensor[T](bs, inter)
	core.SliceFromFloat32(hF, h.Data)

	_, downPost, err := dense.ForwardWebGPU(l.Down, h)
	if err != nil {
		return nil, nil, fmt.Errorf("swiglu Down: %w", err)
	}
	pre = unflatten(h, lay, inter)
	post = unflatten(downPost, lay, l.Cfg.InputDim)
	return pre, post, nil
}

// BackwardWebGPU — the recompute/combine path stays on host, but every dense
// projection it calls (Gate/Up/Down Forward+Backward) still dispatches to
// on-device GEMV/GEMVT because syncProjExec propagates Exec.Backend =
// BackendWebGPU to the children; only the SiLU-derivative elementwise combine
// runs on host. Suite note: "SiLU⊙ device (fwd); proj GEMV/GEMVT device (fwd+bwd);
// combine host (bwd)".
func BackwardWebGPU[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("swiglu: BackendWebGPU but no device (no host fake)")
	}
	l.syncProjExec()
	return backwardHost(l, gradOut, input, pre)
}
