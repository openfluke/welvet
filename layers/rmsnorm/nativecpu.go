package rmsnorm

import (
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/weights"
)

func ForwardNativeCPU[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if l == nil || l.Gamma == nil || l.Gamma.Format != quant.FormatNone || !nativecpp.NativeCPUAvailable() {
		return ForwardCPUTiled(l, input)
	}
	lay, err := parseLayout(l.Cfg.Dim, input)
	if err != nil {
		return nil, nil, err
	}
	g, err := weights.MatrixF32(l.Gamma)
	if err != nil {
		return ForwardCPUTiled(l, input)
	}
	n := lay.batch * lay.seq * lay.dim
	xF := core.SliceAsFloat32(input.Data)
	preF := make([]float32, n)
	postF := make([]float32, n)
	if err := nativecpp.RMSNormForwardF32(
		int32(lay.batch), int32(lay.seq), int32(lay.dim), l.Cfg.Eps,
		int32(l.Gamma.DType), nativecpp.CABBackendCPU,
		g, xF, preF, postF,
	); err != nil {
		return ForwardCPUTiled(l, input)
	}
	preFlat := make([]T, n)
	postFlat := make([]T, n)
	core.SliceFromFloat32(preF, preFlat)
	core.SliceFromFloat32(postF, postFlat)
	return reshapeOut(preFlat, lay), reshapeOut(postFlat, lay), nil
}

func BackwardNativeCPU[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if l == nil || l.Gamma == nil || l.Gamma.Format != quant.FormatNone || !nativecpp.NativeCPUAvailable() {
		return BackwardCPUTiled(l, gradOut, input, pre)
	}
	lay, err := parseLayout(l.Cfg.Dim, input)
	if err != nil {
		return nil, nil, err
	}
	g, err := weights.MatrixF32(l.Gamma)
	if err != nil {
		return BackwardCPUTiled(l, gradOut, input, pre)
	}
	n := lay.batch * lay.seq * lay.dim
	dim := lay.dim
	xF := core.SliceAsFloat32(input.Data)
	gF := core.SliceAsFloat32(gradOut.Data)
	preF := make([]float32, n)
	postF := make([]float32, n)
	dX := make([]float32, n)
	dW := make([]float32, dim)
	gAfter := make([]float32, dim)
	if _, err := nativecpp.RMSNormNormalBPF32(
		int32(lay.batch), int32(lay.seq), int32(lay.dim), l.Cfg.Eps,
		int32(l.Gamma.DType), nativecpp.CABBackendCPU,
		g, xF, gF, false, 0, preF, postF, dX, dW, gAfter,
	); err != nil {
		return BackwardCPUTiled(l, gradOut, input, pre)
	}
	gradInT := reshapeOut(make([]T, n), lay)
	gradW = core.NewTensor[T](1, dim)
	core.SliceFromFloat32(dX, gradInT.Data)
	core.SliceFromFloat32(dW, gradW.Data)
	return gradInT, gradW, nil
}
