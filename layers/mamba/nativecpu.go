package mamba

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
		return nil, nil, fmt.Errorf("mamba: nil native input")
	}
	if !nativecpp.NativeCPUAvailable() || (l.InProj != nil && l.InProj.Weights != nil && l.InProj.Weights.Format != quant.FormatNone) {
		return host()
	}

	if l == nil || l.InProj == nil || l.OutProj == nil || input == nil {
		return nil, nil, fmt.Errorf("mamba: NativeCPU nil")
	}
	if l.InProj.Weights == nil || l.InProj.Weights.Format != quant.FormatNone || !nativecpp.NativeCPUAvailable() {
		return nil, nil, fmt.Errorf("mamba: NativeCPU needs FormatNone + libwelvet")
	}
	lay, err := parseIn(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	packed, err := packMambaNative(l)
	if err != nil {
		return nil, nil, err
	}
	n := lay.batch * lay.t * lay.d
	nw := len(packed)
	xF := core.SliceAsFloat32(input.Data)
	preF := make([]float32, n)
	postF := make([]float32, n)

	if !back {
		err = nativecpp.MambaForwardF32(
			int32(lay.batch), int32(lay.t), int32(l.Cfg.DModel), int32(l.Cfg.DState), int32(l.Cfg.Expand),
			int32(l.InProj.Weights.DType), nativecpp.CABBackendCPU,
			packed, xF, preF, postF,
		)
		if err != nil {
			return nil, nil, err
		}
		pre := core.NewTensor[T](([]int{lay.batch, lay.t, lay.d})...)
		post := core.NewTensor[T](([]int{lay.batch, lay.t, lay.d})...)
		core.SliceFromFloat32(preF, pre.Data)
		core.SliceFromFloat32(postF, post.Data)
		return pre, post, nil
	}
	if gradOut == nil || len(gradOut.Data) != len(postF) {
		return nil, nil, fmt.Errorf("mamba: native gradient shape mismatch")
	}
	gF := core.SliceAsFloat32(gradOut.Data)
	dX := make([]float32, n)
	dW := make([]float32, nw)
	wAfter := make([]float32, nw)
	_, err = nativecpp.MambaNormalBPF32(
		int32(lay.batch), int32(lay.t), int32(l.Cfg.DModel), int32(l.Cfg.DState), int32(l.Cfg.Expand),
		int32(l.InProj.Weights.DType), nativecpp.CABBackendCPU,
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
