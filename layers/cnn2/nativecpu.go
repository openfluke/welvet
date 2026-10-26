package cnn2

import (
	"fmt"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/quant"
)

// ForwardNativeCPU calls the C ABI for the supported FormatNone baseline.
func ForwardNativeCPU[T core.Numeric](l *Layer, input *core.Tensor[T]) (*core.Tensor[T], *core.Tensor[T], error) {
	return nativePass(l, input, nil, nil, false)
}

// BackwardNativeCPU returns C++ dInput/dWeights without updating parameters.
func BackwardNativeCPU[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (*core.Tensor[T], *core.Tensor[T], error) {
	return nativePass(l, input, gradOut, pre, true)
}
func nativePass[T core.Numeric](l *Layer, input, gradOut, savedPre *core.Tensor[T], back bool) (*core.Tensor[T], *core.Tensor[T], error) {
	host := func() (*core.Tensor[T], *core.Tensor[T], error) {
		if back {
			return BackwardCPUTiled(l, gradOut, input, savedPre)
		}
		return ForwardCPUTiled(l, input)
	}
	if l == nil || input == nil {
		return nil, nil, fmt.Errorf("cnn2: nil native input")
	}
	if !nativecpp.NativeCPUAvailable() || (l.Proj != nil && l.Proj.Weights != nil && l.Proj.Weights.Format != quant.FormatNone) {
		return host()
	}

	if l == nil || l.Proj == nil || l.Proj.Weights == nil || input == nil {
		return nil, nil, fmt.Errorf("cnn2: NativeCPU nil")
	}
	if l.Proj.Weights.Format != quant.FormatNone || !nativecpp.NativeCPUAvailable() {
		return nil, nil, fmt.Errorf("cnn2: NativeCPU needs FormatNone + libwelvet")
	}
	lay, err := parseLayout(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	w, err := l.Proj.NativeMatrixF32()
	if err != nil {
		return nil, nil, err
	}
	sp := lay.outH * lay.outW
	nOut := lay.batch * lay.filters * sp
	nIn := lay.batch * lay.inC * lay.inH * lay.inW
	nw := len(w)
	xF := core.SliceAsFloat32(input.Data)
	preF := make([]float32, nOut)
	postF := make([]float32, nOut)

	if !back {
		err = nativecpp.CNN2ForwardF32(
			int32(lay.batch), int32(lay.inC), int32(lay.filters), int32(lay.inH), int32(lay.inW),
			int32(l.Cfg.Kernel), int32(l.Cfg.Stride), int32(l.Cfg.Padding),
			int32(l.Cfg.Activation), int32(l.Proj.Weights.DType), nativecpp.CABBackendCPU,
			w, l.Proj.Weights.Bias, xF, preF, postF,
		)
		if err != nil {
			return nil, nil, err
		}
		pre := core.NewTensor[T](([]int{lay.batch, lay.filters, lay.outH, lay.outW})...)
		post := core.NewTensor[T](([]int{lay.batch, lay.filters, lay.outH, lay.outW})...)
		core.SliceFromFloat32(preF, pre.Data)
		core.SliceFromFloat32(postF, post.Data)
		return pre, post, nil
	}
	if gradOut == nil || len(gradOut.Data) != len(postF) {
		return nil, nil, fmt.Errorf("cnn2: native gradient shape mismatch")
	}
	gF := core.SliceAsFloat32(gradOut.Data)
	dX := make([]float32, nIn)
	dW := make([]float32, nw)
	wAfter := make([]float32, nw)
	_, err = nativecpp.CNN2NormalBPF32(
		int32(lay.batch), int32(lay.inC), int32(lay.filters), int32(lay.inH), int32(lay.inW),
		int32(l.Cfg.Kernel), int32(l.Cfg.Stride), int32(l.Cfg.Padding),
		int32(l.Cfg.Activation), int32(l.Proj.Weights.DType), nativecpp.CABBackendCPU,
		w, l.Proj.Weights.Bias, xF, gF, false, 0,
		preF, postF, dX, dW, wAfter,
	)

	if err != nil {
		return nil, nil, err
	}
	gx := core.NewTensor[T](input.Shape...)
	dw := core.NewTensor[T](l.Proj.Weights.Rows, l.Proj.Weights.Cols)
	core.SliceFromFloat32(dX, gx.Data)
	core.SliceFromFloat32(dW, dw.Data)
	return gx, dw, nil
}

// ApplyExecConfig resolves a nativecpp.ExecConfig onto this layer's Exec.Backend.
func (l *Layer) ApplyExecConfig(cfg nativecpp.ExecConfig, volume, sector, layerKey string) {
	if l == nil {
		return
	}
	nativecpp.ApplyExecConfig(cfg, volume, sector, layerKey, &l.Exec)
}
