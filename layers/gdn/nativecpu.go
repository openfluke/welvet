package gdn

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
			return BackwardSIMD(l, gradOut, input, savedPre)
		}
		return forwardHost(l, input)
	}
	if l == nil || input == nil {
		return nil, nil, fmt.Errorf("gdn: nil native input")
	}
	if !nativecpp.NativeCPUAvailable() || (l.InQKV != nil && l.InQKV.Format != quant.FormatNone) {
		return host()
	}
	if l.HasState || (l.Cfg.Eps != 0 && l.Cfg.Eps != 1e-6) {
		return host()
	}
	if l == nil || input == nil {
		return nil, nil, fmt.Errorf("gdn: NativeCPU nil")
	}
	if l.InQKV == nil || l.InQKV.Format != quant.FormatNone || !nativecpp.NativeCPUAvailable() {
		return nil, nil, fmt.Errorf("gdn: NativeCPU needs FormatNone + libwelvet")
	}
	if len(input.Shape) != 3 || input.Shape[2] != l.Cfg.HiddenSize {
		return nil, nil, fmt.Errorf("gdn: NativeCPU need [batch,seq,hidden]")
	}
	batch, seq, h := input.Shape[0], input.Shape[1], l.Cfg.HiddenSize
	packed, err := packGDNNative(l)
	if err != nil {
		return nil, nil, err
	}
	n := batch * seq * h
	nw := len(packed)
	xF := core.SliceAsFloat32(input.Data)
	preF := make([]float32, n)
	postF := make([]float32, n)
	c := l.Cfg
	if !back {
		err = nativecpp.GDNForwardF32(
			int32(batch), int32(seq), int32(h),
			int32(c.NumKeyHeads), int32(c.NumValueHeads),
			int32(c.KeyHeadDim), int32(c.ValueHeadDim), int32(c.ConvKernel),
			int32(core.DTypeFloat32), nativecpp.CABBackendCPU,
			packed, xF, preF, postF,
		)
		if err != nil {
			return nil, nil, err
		}
		pre := core.NewTensor[T]((input.Shape)...)
		post := core.NewTensor[T]((input.Shape)...)
		core.SliceFromFloat32(preF, pre.Data)
		core.SliceFromFloat32(postF, post.Data)
		return pre, post, nil
	}
	if gradOut == nil || len(gradOut.Data) != len(postF) {
		return nil, nil, fmt.Errorf("gdn: native gradient shape mismatch")
	}
	gF := core.SliceAsFloat32(gradOut.Data)
	dX := make([]float32, n)
	dW := make([]float32, nw)
	wAfter := make([]float32, nw)

	_, err = nativecpp.GDNNormalBPF32(
		int32(batch), int32(seq), int32(h),
		int32(c.NumKeyHeads), int32(c.NumValueHeads),
		int32(c.KeyHeadDim), int32(c.ValueHeadDim), int32(c.ConvKernel),
		int32(core.DTypeFloat32), nativecpp.CABBackendCPU,
		packed, xF, gF, false, 0,
		preF, postF, dX, dW, wAfter,
	)

	if err != nil {
		return nil, nil, err
	}
	gx := core.NewTensor[T](input.Shape...)
	dw := core.NewTensor[T](nw)
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
