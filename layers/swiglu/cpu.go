package swiglu

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/simd"
)

// ForwardCPUTiled — Gate/Up/Down via dense; SiLU ⊙ on host.
func ForwardCPUTiled[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	return forwardHost(l, input, false)
}

// BackwardCPUTiled — reverse SwiGLU; gradW = concat(dGate, dUp, dDown).
func BackwardCPUTiled[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	return backwardHost(l, gradOut, input, pre, false)
}

func forwardHost[T core.Numeric](l *Layer, input *core.Tensor[T], useSIMD bool) (pre, post *core.Tensor[T], err error) {
	lay, err := parseLayout(l.Cfg.InputDim, input)
	if err != nil {
		return nil, nil, err
	}
	flat := flatten(input, lay)
	inter := l.Cfg.IntermediateDim

	gatePre, _, err := dense.Forward(l.Gate, flat)
	if err != nil {
		return nil, nil, fmt.Errorf("swiglu Gate: %w", err)
	}
	_, upPost, err := dense.Forward(l.Up, flat)
	if err != nil {
		return nil, nil, fmt.Errorf("swiglu Up: %w", err)
	}

	bs := lay.batch * lay.seq
	h := core.NewTensor[T](bs, inter)
	n := bs * inter
	if useSIMD && simd.Enabled() {
		gateF := core.SliceAsFloat32(gatePre.Data)
		upF := core.SliceAsFloat32(upPost.Data)
		hF := make([]float32, n)
		simd.SiluMulF32(gateF, upF, hF, n)
		core.SliceFromFloat32(hF, h.Data)
	} else {
		for i := 0; i < n; i++ {
			g := core.Activate(gatePre.Data[i], core.ActivationSilu)
			h.Data[i] = core.FromFloat64[T](core.AsFloat64(g) * core.AsFloat64(upPost.Data[i]))
		}
	}

	downPre, downPost, err := dense.Forward(l.Down, h)
	if err != nil {
		return nil, nil, fmt.Errorf("swiglu Down: %w", err)
	}
	_ = downPre
	pre = unflatten(h, lay, inter)
	post = unflatten(downPost, lay, l.Cfg.InputDim)
	return pre, post, nil
}

func backwardHost[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T], useSIMD bool) (gradIn, gradW *core.Tensor[T], err error) {
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

	dGate := core.NewTensor[T](bs, inter)
	dUp := core.NewTensor[T](bs, inter)
	n := bs * inter
	if useSIMD && simd.Enabled() {
		gateF := core.SliceAsFloat32(gatePre.Data)
		upF := core.SliceAsFloat32(upPost.Data)
		gradHF := core.SliceAsFloat32(gradH.Data)
		dGateF := make([]float32, n)
		dUpF := make([]float32, n)
		simd.SiluMulBwdF32(gateF, upF, gradHF, dGateF, dUpF, n)
		core.SliceFromFloat32(dGateF, dGate.Data)
		core.SliceFromFloat32(dUpF, dUp.Data)
	} else {
		for i := 0; i < n; i++ {
			silu := core.Activate(gatePre.Data[i], core.ActivationSilu)
			dsilu := core.ActivateDeriv(gatePre.Data[i], core.ActivationSilu)
			dh := core.AsFloat64(gradH.Data[i])
			dUp.Data[i] = core.FromFloat64[T](dh * core.AsFloat64(silu))
			dGate.Data[i] = core.FromFloat64[T](dh * core.AsFloat64(upPost.Data[i]) * core.AsFloat64(dsilu))
		}
	}

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

func copyGrad[T core.Numeric](dst *core.Tensor[T], off int, src *core.Tensor[T]) int {
	n := src.Len()
	copy(dst.Data[off:off+n], src.Data)
	return off + n
}
