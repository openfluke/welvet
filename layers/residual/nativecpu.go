package residual

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
		return nil, nil, fmt.Errorf("residual: nil native input")
	}
	if !nativecpp.NativeCPUAvailable() || (len(l.Children) > 0 && l.Children[0] != nil && l.Children[0].Weights != nil && l.Children[0].Weights.Format != quant.FormatNone) {
		return host()
	}
	if len(l.Ops) > 0 || !wholePlacement(l) {
		return host()
	}
	for _, ch := range l.Children {
		if ch == nil || ch.Weights == nil {
			return nil, nil, fmt.Errorf("native CPU: nil child")
		}
		if len(ch.Weights.Bias) > 0 || ch.Core.Activation != l.Children[0].Core.Activation || ch.Weights.DType != l.Children[0].Weights.DType {
			return host()
		}
	}
	if l == nil || input == nil {
		return nil, nil, fmt.Errorf("residual: NativeCPU nil")
	}
	if len(l.Ops) > 0 {
		return nil, nil, fmt.Errorf("residual: NativeCPU needs Dense Children (not mixed Ops)")
	}
	if len(l.Children) == 0 || l.Children[0] == nil || l.Children[0].Weights == nil {
		return nil, nil, fmt.Errorf("residual: NativeCPU nil children")
	}
	if l.Children[0].Weights.Format != quant.FormatNone || !nativecpp.NativeCPUAvailable() {
		return nil, nil, fmt.Errorf("residual: NativeCPU needs FormatNone + libwelvet")
	}
	lay, err := parseLayout(l.Cfg.Dim, input)
	if err != nil {
		return nil, nil, err
	}
	packed, _, err := packDenseChildren(l.Children)
	if err != nil {
		return nil, nil, err
	}
	dim, depth := l.Cfg.Dim, len(l.Children)
	n := lay.batch * lay.seq * dim
	nw := len(packed)
	xF := core.SliceAsFloat32(input.Data)
	preF := make([]float32, n)
	postF := make([]float32, n)
	childAct := int32(l.Children[0].Core.Activation)
	if !back {
		err = nativecpp.ResidualForwardF32(
			int32(lay.batch), int32(lay.seq), int32(dim), int32(depth), childAct,
			int32(l.Children[0].Weights.DType), nativecpp.CABBackendCPU,
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
		return nil, nil, fmt.Errorf("residual: native gradient shape mismatch")
	}
	gF := core.SliceAsFloat32(gradOut.Data)
	dX := make([]float32, n)
	dW := make([]float32, nw)
	wAfter := make([]float32, nw)

	_, err = nativecpp.ResidualNormalBPF32(
		int32(lay.batch), int32(lay.seq), int32(dim), int32(depth), childAct,
		int32(l.Children[0].Weights.DType), nativecpp.CABBackendCPU,
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

// ApplyExecConfig resolves a nativecpp.ExecConfig onto this layer's Exec and syncs children.
func (l *Layer) ApplyExecConfig(cfg nativecpp.ExecConfig, volume, sector, layerKey string) {
	if l == nil {
		return
	}
	nativecpp.ApplyExecConfig(cfg, volume, sector, layerKey, &l.Exec)
	l.syncChildExec()
}
