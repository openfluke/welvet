package embedding

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
		return nil, nil, fmt.Errorf("embedding: nil native input")
	}
	if !nativecpp.NativeCPUAvailable() || (l.Weights != nil && l.Weights.Format != quant.FormatNone) {
		return host()
	}

	if l == nil || l.Weights == nil || input == nil {
		return nil, nil, fmt.Errorf("embedding: NativeCPU nil")
	}
	if l.Weights.Format != quant.FormatNone || !nativecpp.NativeCPUAvailable() {
		return nil, nil, fmt.Errorf("embedding: NativeCPU needs FormatNone + libwelvet")
	}
	lay, err := parseLayout(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	w, err := l.NativeMatrixF32()
	if err != nil {
		return nil, nil, err
	}
	nTok := lay.batch * lay.seq
	nOut := nTok * lay.emb
	nw := lay.vocab * lay.emb
	xF := core.SliceAsFloat32(input.Data)
	preF := make([]float32, nOut)
	postF := make([]float32, nOut)

	if !back {
		err = nativecpp.EmbeddingForwardF32(
			int32(lay.batch), int32(lay.vocab), int32(lay.emb), int32(lay.seq),
			int32(lay.inRank), int32(lay.inLast),
			int32(l.Core.Activation), int32(l.Weights.DType), nativecpp.CABBackendCPU,
			w, xF, preF, postF,
		)
		if err != nil {
			return nil, nil, err
		}
		pre := core.NewTensor[T]((outShape(lay))...)
		post := core.NewTensor[T]((outShape(lay))...)
		core.SliceFromFloat32(preF, pre.Data)
		core.SliceFromFloat32(postF, post.Data)
		return pre, post, nil
	}
	if gradOut == nil || len(gradOut.Data) < len(postF) {
		return nil, nil, fmt.Errorf("embedding: native gradient shape mismatch")
	}
	gF := core.SliceAsFloat32(gradOut.Data)
	dX := make([]float32, nTok*lay.inLast)
	dW := make([]float32, nw)
	wAfter := make([]float32, nw)
	_, err = nativecpp.EmbeddingNormalBPF32(
		int32(lay.batch), int32(lay.vocab), int32(lay.emb), int32(lay.seq),
		int32(lay.inRank), int32(lay.inLast),
		int32(l.Core.Activation), int32(l.Weights.DType), nativecpp.CABBackendCPU,
		w, xF, gF, false, 0,
		preF, postF, dX, dW, wAfter,
	)

	if err != nil {
		return nil, nil, err
	}
	gx := core.NewTensor[T](input.Shape...)
	dw := core.NewTensor[T](l.Weights.Rows, l.Weights.Cols)
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
