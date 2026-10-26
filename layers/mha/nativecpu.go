package mha

import (
	"fmt"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
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
		return nil, nil, fmt.Errorf("mha: nil native input")
	}
	if !nativecpp.NativeCPUAvailable() || (l.Q != nil && l.Q.Weights != nil && l.Q.Weights.Format != quant.FormatNone) {
		return host()
	}
	c := l.Cfg
	if c.Mode != ModeSelf || c.QKNorm || c.Dropout != 0 || c.ScaleOverride != 0 || (c.Mask != MaskCausal && c.Mask != MaskBidirectional) || (c.Pos != PosNone && c.Pos != PosRoPE) || l.KVOffset != 0 || l.CustomAllow != nil || (c.RoPETheta != 0 && c.RoPETheta != 10000) {
		return host()
	}
	if l == nil || l.Q == nil || l.K == nil || l.V == nil || l.O == nil || input == nil {
		return nil, nil, fmt.Errorf("mha: NativeCPU nil")
	}
	if l.Q.Weights == nil || l.Q.Weights.Format != quant.FormatNone || !nativecpp.NativeCPUAvailable() {
		return nil, nil, fmt.Errorf("mha: NativeCPU needs FormatNone + libwelvet")
	}
	if l.Cfg.Mode != ModeSelf {
		return nil, nil, fmt.Errorf("mha: NativeCPU C ABI is ModeSelf only")
	}
	for _, projection := range []*dense.Layer{l.Q, l.K, l.V, l.O} {
		if projection.Weights == nil {
			return nil, nil, fmt.Errorf("mha: nil native projection")
		}
		if projection.Weights.Format != quant.FormatNone || projection.Weights.DType != l.Q.Weights.DType || len(projection.Weights.Bias) > 0 {
			return host()
		}
	}
	lay, err := parseLayout(l.Cfg.DModel, input)
	if err != nil {
		return nil, nil, err
	}
	cfg := l.Cfg
	maxSeq := cfg.MaxSeqLen
	if maxSeq <= 0 {
		maxSeq = 512
	}
	q, err := l.Q.NativeMatrixF32()
	if err != nil {
		return nil, nil, err
	}
	k, err := l.K.NativeMatrixF32()
	if err != nil {
		return nil, nil, err
	}
	v, err := l.V.NativeMatrixF32()
	if err != nil {
		return nil, nil, err
	}
	o, err := l.O.NativeMatrixF32()
	if err != nil {
		return nil, nil, err
	}
	n := lay.batch * lay.seqLen * lay.dModel
	nw := len(q) + len(k) + len(v) + len(o)
	xF := core.SliceAsFloat32(input.Data)
	preF := make([]float32, n)
	postF := make([]float32, n)

	if !back {
		err = nativecpp.MHAForwardF32(
			int32(lay.batch), int32(lay.seqLen), int32(cfg.DModel),
			int32(cfg.NumHeads), int32(cfg.NumKVHeads), int32(cfg.HeadDim), int32(maxSeq),
			int32(cfg.Mask), int32(cfg.Pos),
			int32(l.Q.Weights.DType), nativecpp.CABBackendCPU,
			q, k, v, o, xF, preF, postF,
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
		return nil, nil, fmt.Errorf("mha: native gradient shape mismatch")
	}
	gF := core.SliceAsFloat32(gradOut.Data)
	dX := make([]float32, n)
	dW := make([]float32, nw)
	wAfter := make([]float32, nw)
	_, err = nativecpp.MHANormalBPF32(
		int32(lay.batch), int32(lay.seqLen), int32(cfg.DModel),
		int32(cfg.NumHeads), int32(cfg.NumKVHeads), int32(cfg.HeadDim), int32(maxSeq),
		int32(cfg.Mask), int32(cfg.Pos),
		int32(l.Q.Weights.DType), nativecpp.CABBackendCPU,
		q, k, v, o, xF, gF, false, 0,
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
