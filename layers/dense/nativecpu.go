package dense

import (
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/weights"
)

// useNativeCPU reports whether this Dense can take the C ABI path (FormatNone).
func useNativeCPU(l *Layer) bool {
	return l != nil && l.Weights != nil &&
		l.Weights.Format == quant.FormatNone &&
		nativecpp.NativeCPUAvailable()
}

// ForwardNativeCPU runs welvet_dense_forward_f32 with backend=CPU (exact_bit).
// Falls back to ForwardCPUTiled when FormatNone is unavailable.
func ForwardNativeCPU[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if !useNativeCPU(l) {
		return ForwardCPUTiled(l, input)
	}
	batch, in, out, err := dims(l, input)
	if err != nil {
		return nil, nil, err
	}
	w, err := l.NativeMatrixF32()
	if err != nil {
		return nil, nil, err
	}
	xF := core.SliceAsFloat32(input.Data)
	preF := make([]float32, batch*out)
	postF := make([]float32, batch*out)
	if err := nativecpp.DenseForwardF32(
		int32(batch), int32(in), int32(out),
		int32(l.Core.Activation), int32(l.Weights.DType), nativecpp.CABBackendCPU,
		w, l.Weights.Bias, xF, preF, postF,
	); err != nil {
		return nil, nil, err
	}
	pre = core.NewTensor[T](batch, out)
	post = core.NewTensor[T](batch, out)
	core.SliceFromFloat32(preF, pre.Data)
	core.SliceFromFloat32(postF, post.Data)
	return pre, post, nil
}

// BackwardNativeCPU uses welvet_dense_normalbp_f32 (use_target=0, lr=0) for dX/dW.
func BackwardNativeCPU[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if !useNativeCPU(l) {
		return BackwardCPUTiled(l, gradOut, input, pre)
	}
	_ = pre // C NormalBP recomputes forward + act-deriv internally
	batch, in, out, err := dims(l, input)
	if err != nil {
		return nil, nil, err
	}
	w, err := weights.MatrixF32(l.Weights)
	if err != nil {
		return nil, nil, err
	}
	xF := core.SliceAsFloat32(input.Data)
	gF := core.SliceAsFloat32(gradOut.Data)
	preF := make([]float32, batch*out)
	postF := make([]float32, batch*out)
	dX := make([]float32, batch*in)
	dW := make([]float32, out*in)
	wAfter := make([]float32, out*in)
	if _, err := nativecpp.DenseNormalBPF32(
		int32(batch), int32(in), int32(out),
		int32(l.Core.Activation), int32(l.Weights.DType), nativecpp.CABBackendCPU,
		w, l.Weights.Bias, xF, gF, false, 0,
		preF, postF, dX, dW, wAfter,
	); err != nil {
		return nil, nil, err
	}
	gradIn = core.NewTensor[T](batch, in)
	gradW = core.NewTensor[T](out, in)
	core.SliceFromFloat32(dX, gradIn.Data)
	core.SliceFromFloat32(dW, gradW.Data)
	return gradIn, gradW, nil
}

// ApplyExecConfig resolves a nativecpp.ExecConfig onto this layer's Exec.Backend.
func (l *Layer) ApplyExecConfig(cfg nativecpp.ExecConfig, volume, sector, layerKey string) {
	if l == nil {
		return
	}
	nativecpp.ApplyExecConfig(cfg, volume, sector, layerKey, &l.Exec)
}

// SetBackend sets Exec.Backend (convenience for tests / planners).
func (l *Layer) SetBackend(b core.Backend) {
	if l != nil {
		l.Exec.Backend = b
		l.Exec.BackendPinned = true
	}
}
