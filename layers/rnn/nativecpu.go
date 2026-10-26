package rnn

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
		return nil, nil, fmt.Errorf("rnn: nil native input")
	}
	if !nativecpp.NativeCPUAvailable() || (l.IH != nil && l.IH.Weights != nil && l.IH.Weights.Format != quant.FormatNone) {
		return host()
	}

	if l == nil || l.IH == nil || l.HH == nil || input == nil {
		return nil, nil, fmt.Errorf("rnn: NativeCPU nil")
	}
	if l.IH.Weights == nil || l.IH.Weights.Format != quant.FormatNone || !nativecpp.NativeCPUAvailable() {
		return nil, nil, fmt.Errorf("rnn: NativeCPU needs FormatNone + libwelvet")
	}
	lay, err := parseLayout(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	packed, err := packRNNNative(l)
	if err != nil {
		return nil, nil, err
	}
	nOut := lay.batch * lay.seq * lay.hid
	nIn := lay.batch * lay.seq * lay.in
	nw := len(packed)
	xF := core.SliceAsFloat32(input.Data)
	preF := make([]float32, nOut)
	postF := make([]float32, nOut)

	if !back {
		err = nativecpp.RNNForwardF32(
			int32(lay.batch), int32(lay.in), int32(lay.hid), int32(lay.seq),
			int32(l.IH.Weights.DType), nativecpp.CABBackendCPU,
			packed, xF, preF, postF,
		)
		if err != nil {
			return nil, nil, err
		}
		pre := core.NewTensor[T](([]int{lay.batch, lay.seq, lay.hid})...)
		post := core.NewTensor[T](([]int{lay.batch, lay.seq, lay.hid})...)
		core.SliceFromFloat32(preF, pre.Data)
		core.SliceFromFloat32(postF, post.Data)
		return pre, post, nil
	}
	if gradOut == nil || len(gradOut.Data) != len(postF) {
		return nil, nil, fmt.Errorf("rnn: native gradient shape mismatch")
	}
	gF := core.SliceAsFloat32(gradOut.Data)
	dX := make([]float32, nIn)
	dW := make([]float32, nw)
	wAfter := make([]float32, nw)
	_, err = nativecpp.RNNNormalBPF32(
		int32(lay.batch), int32(lay.in), int32(lay.hid), int32(lay.seq),
		int32(l.IH.Weights.DType), nativecpp.CABBackendCPU,
		packed, xF, gF, false, 0,
		preF, postF, dX, dW, wAfter,
	)

	if err != nil {
		return nil, nil, err
	}
	gx := core.NewTensor[T](input.Shape...)
	dw := core.NewTensor[T](1, nw)
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
