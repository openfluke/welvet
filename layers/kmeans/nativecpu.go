package kmeans

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
		return nil, nil, fmt.Errorf("kmeans: nil native input")
	}
	if !nativecpp.NativeCPUAvailable() || (l.Centers != nil && l.Centers.Weights != nil && l.Centers.Weights.Format != quant.FormatNone) {
		return host()
	}

	if l == nil || l.Centers == nil || l.Centers.Weights == nil || input == nil {
		return nil, nil, fmt.Errorf("kmeans: NativeCPU nil")
	}
	if l.Centers.Weights.Format != quant.FormatNone || !nativecpp.NativeCPUAvailable() {
		return nil, nil, fmt.Errorf("kmeans: NativeCPU needs FormatNone + libwelvet")
	}
	lay, err := parseIn(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	w, err := l.Centers.NativeMatrixF32()
	if err != nil {
		return nil, nil, err
	}
	batch, seq := lay.batch, 1
	if len(lay.shape) == 3 {
		batch, seq = lay.shape[0], lay.shape[1]
	}
	outDim := l.Cfg.OutDim()
	nOut := batch * seq * outDim
	nIn := batch * seq * lay.d
	nw := len(w)
	xF := core.SliceAsFloat32(input.Data)
	preF := make([]float32, nOut)
	postF := make([]float32, nOut)

	if !back {
		err = nativecpp.KMeansForwardF32(int32(len(xF)), int32(lay.d), int32(lay.k), outputModeCAB(l.Cfg.OutputMode), int32(l.Cfg.Activation), l.Cfg.temp(), int32(l.Centers.Weights.DType), nativecpp.CABBackendCPU, int32(len(input.Shape)), nativeShape(input.Shape), w, xF, preF, postF)
		if err != nil {
			return nil, nil, err
		}
		pre := core.NewTensor[T]((append(append([]int(nil), input.Shape[:len(input.Shape)-1]...), outDim))...)
		post := core.NewTensor[T]((append(append([]int(nil), input.Shape[:len(input.Shape)-1]...), outDim))...)
		core.SliceFromFloat32(preF, pre.Data)
		core.SliceFromFloat32(postF, post.Data)
		return pre, post, nil
	}
	if gradOut == nil || len(gradOut.Data) != len(postF) {
		return nil, nil, fmt.Errorf("kmeans: native gradient shape mismatch")
	}
	gF := core.SliceAsFloat32(gradOut.Data)
	dX := make([]float32, nIn)
	dW := make([]float32, nw)
	wAfter := make([]float32, nw)
	_, err = nativecpp.KMeansNormalBPF32(
		int32(batch), int32(seq), int32(lay.k), int32(lay.d),
		outputModeCAB(l.Cfg.OutputMode), int32(l.Cfg.Activation), l.Cfg.temp(),
		int32(l.Centers.Weights.DType), nativecpp.CABBackendCPU,
		w, xF, gF, false, 0,
		preF, postF, dX, dW, wAfter,
	)

	if err != nil {
		return nil, nil, err
	}
	gx := core.NewTensor[T](input.Shape...)
	dw := core.NewTensor[T](l.Centers.Weights.Rows, l.Centers.Weights.Cols)
	core.SliceFromFloat32(dX, gx.Data)
	core.SliceFromFloat32(dW, dw.Data)
	return gx, dw, nil
}
func nativeShape(shape []int) []int32 {
	out := make([]int32, len(shape))
	for i, v := range shape {
		out[i] = int32(v)
	}
	return out
}

// ApplyExecConfig resolves a nativecpp.ExecConfig onto this layer's Exec.Backend.
func (l *Layer) ApplyExecConfig(cfg nativecpp.ExecConfig, volume, sector, layerKey string) {
	if l == nil {
		return
	}
	nativecpp.ApplyExecConfig(cfg, volume, sector, layerKey, &l.Exec)
}
