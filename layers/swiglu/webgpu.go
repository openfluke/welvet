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

// BackwardWebGPU — Gate/Up/Down projections run via dense.Backward (on-device
// GEMVT when Exec.Backend=BackendWebGPU); the SiLU(gate)⊙up derivative combine
// runs on-device via webgpu.SwiGLUFuseBackward after Down backward yields gradH.
func BackwardWebGPU[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("swiglu: BackendWebGPU but no device (no host fake)")
	}
	l.syncProjExec()
	return backwardWebGPU(l, gradOut, input, pre)
}

func backwardWebGPU[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	lay, err := parseLayout(l.Cfg.InputDim, input)
	if err != nil {
		return nil, nil, err
	}
	inter := l.Cfg.IntermediateDim
	bs := lay.batch * lay.seq
	flat := flatten(input, lay)
	gy := flatten(gradOut, layout{batch: lay.batch, seq: lay.seq, in: l.Cfg.InputDim, elemStride: lay.seq * l.Cfg.InputDim})

	if pre == nil || pre.Len() < bs*inter {
		return nil, nil, fmt.Errorf("swiglu: pre (h) missing or wrong size")
	}
	h := core.NewTensor[T](bs, inter)
	copy(h.Data, pre.Data[:bs*inter])

	downPre, _, err := dense.Forward(l.Down, h)
	if err != nil {
		return nil, nil, fmt.Errorf("swiglu Down recompute: %w", err)
	}
	gradH, gradWD, err := dense.Backward(l.Down, gy, h, downPre)
	if err != nil {
		return nil, nil, fmt.Errorf("swiglu Down bwd: %w", err)
	}

	gatePre, _, err := dense.Forward(l.Gate, flat)
	if err != nil {
		return nil, nil, fmt.Errorf("swiglu Gate recompute: %w", err)
	}
	upPre, upPost, err := dense.Forward(l.Up, flat)
	if err != nil {
		return nil, nil, fmt.Errorf("swiglu Up recompute: %w", err)
	}

	gateF := core.SliceAsFloat32(gatePre.Data)
	upF := core.SliceAsFloat32(upPost.Data)
	gradHF := core.SliceAsFloat32(gradH.Data)
	dGateF := make([]float32, bs*inter)
	dUpF := make([]float32, bs*inter)
	if err := webgpu.SwiGLUFuseBackward(gateF, upF, gradHF, dGateF, dUpF, bs*inter); err != nil {
		return nil, nil, err
	}
	dGate := core.NewTensor[T](bs, inter)
	dUp := core.NewTensor[T](bs, inter)
	core.SliceFromFloat32(dGateF, dGate.Data)
	core.SliceFromFloat32(dUpF, dUp.Data)

	gradInG, gradWG, err := dense.Backward(l.Gate, dGate, flat, gatePre)
	if err != nil {
		return nil, nil, fmt.Errorf("swiglu Gate bwd: %w", err)
	}
	gradInU, gradWU, err := dense.Backward(l.Up, dUp, flat, upPre)
	if err != nil {
		return nil, nil, fmt.Errorf("swiglu Up bwd: %w", err)
	}

	gradInFlat := core.NewTensor[T](bs, l.Cfg.InputDim)
	for i := 0; i < bs*l.Cfg.InputDim; i++ {
		sum := core.AsFloat64(gradInG.Data[i]) + core.AsFloat64(gradInU.Data[i])
		gradInFlat.Data[i] = core.FromFloat64[T](sum)
	}
	gradIn = unflatten(gradInFlat, lay, l.Cfg.InputDim)

	gradW = core.NewTensor[T](l.GradWSize())
	off := 0
	off = copyGrad(gradW, off, gradWG)
	off = copyGrad(gradW, off, gradWU)
	_ = copyGrad(gradW, off, gradWD)
	return gradIn, gradW, nil
}
