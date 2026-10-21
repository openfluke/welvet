package swiglu

import (
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/weights"
)

// ForwardNativeCPU calls welvet_swiglu_forward_f32 for FormatNone projections.
func ForwardNativeCPU[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	l.syncProjExec()
	if l == nil || l.Gate == nil || l.Up == nil || l.Down == nil ||
		l.Gate.Weights == nil || l.Gate.Weights.Format != quant.FormatNone ||
		!nativecpp.NativeCPUAvailable() {
		return ForwardCPUTiled(l, input)
	}
	lay, err := parseLayout(l.Cfg.InputDim, input)
	if err != nil {
		return nil, nil, err
	}
	gate, err := weights.MatrixF32(l.Gate.Weights)
	if err != nil {
		return ForwardCPUTiled(l, input)
	}
	up, err := weights.MatrixF32(l.Up.Weights)
	if err != nil {
		return ForwardCPUTiled(l, input)
	}
	down, err := weights.MatrixF32(l.Down.Weights)
	if err != nil {
		return ForwardCPUTiled(l, input)
	}
	xF := core.SliceAsFloat32(input.Data)
	inter := l.Cfg.IntermediateDim
	in := l.Cfg.InputDim
	bs := lay.batch * lay.seq
	preF := make([]float32, bs*inter)
	postF := make([]float32, bs*in)
	if err := nativecpp.SwiGLUForwardF32(
		int32(lay.batch), int32(lay.seq), int32(in), int32(inter),
		int32(l.Gate.Weights.DType), nativecpp.CABBackendCPU,
		gate, up, down, xF, preF, postF,
	); err != nil {
		return ForwardCPUTiled(l, input)
	}
	h := core.NewTensor[T](bs, inter)
	y := core.NewTensor[T](bs, in)
	core.SliceFromFloat32(preF, h.Data)
	core.SliceFromFloat32(postF, y.Data)
	return unflatten(h, lay, inter), unflatten(y, lay, in), nil
}

// BackwardNativeCPU uses welvet_swiglu_normalbp_f32 (lr=0).
func BackwardNativeCPU[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	l.syncProjExec()
	if l == nil || l.Gate == nil || l.Gate.Weights == nil || l.Gate.Weights.Format != quant.FormatNone ||
		!nativecpp.NativeCPUAvailable() {
		return BackwardCPUTiled(l, gradOut, input, pre)
	}
	lay, err := parseLayout(l.Cfg.InputDim, input)
	if err != nil {
		return nil, nil, err
	}
	gate, err := weights.MatrixF32(l.Gate.Weights)
	if err != nil {
		return BackwardCPUTiled(l, gradOut, input, pre)
	}
	up, err := weights.MatrixF32(l.Up.Weights)
	if err != nil {
		return BackwardCPUTiled(l, gradOut, input, pre)
	}
	down, err := weights.MatrixF32(l.Down.Weights)
	if err != nil {
		return BackwardCPUTiled(l, gradOut, input, pre)
	}
	in, inter := l.Cfg.InputDim, l.Cfg.IntermediateDim
	bs := lay.batch * lay.seq
	xF := core.SliceAsFloat32(input.Data)
	gF := core.SliceAsFloat32(gradOut.Data)
	wN := len(gate) + len(up) + len(down)
	preF := make([]float32, bs*inter)
	postF := make([]float32, bs*in)
	dX := make([]float32, bs*in)
	dW := make([]float32, wN)
	wAfter := make([]float32, wN)
	if _, err := nativecpp.SwiGLUNormalBPF32(
		int32(lay.batch), int32(lay.seq), int32(in), int32(inter),
		int32(l.Gate.Weights.DType), nativecpp.CABBackendCPU,
		gate, up, down, xF, gF, false, 0,
		preF, postF, dX, dW, wAfter,
	); err != nil {
		return BackwardCPUTiled(l, gradOut, input, pre)
	}
	gx := core.NewTensor[T](bs, in)
	gw := core.NewTensor[T](wN)
	core.SliceFromFloat32(dX, gx.Data)
	core.SliceFromFloat32(dW, gw.Data)
	return unflatten(gx, lay, in), gw, nil
}
