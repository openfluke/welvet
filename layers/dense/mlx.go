package dense

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/nativemlx"
	"github.com/openfluke/welvet/quant"
)

func (l *Layer) ensureMLXMirror() *nativemlx.WeightMirror {
	if l.mlxMir == nil {
		l.mlxMir = &nativemlx.WeightMirror{}
	}
	return l.mlxMir
}

// SyncToMLX copies Go Weights (+ Bias) into the MLX mirror and checks GPU caps.
// Invalidates any prepared graph — next ForwardMLX will prepare then execute.
func (l *Layer) SyncToMLX() error {
	if l == nil || l.Weights == nil {
		return fmt.Errorf("dense: SyncToMLX nil")
	}
	if l.Weights.Format != quant.FormatNone {
		return fmt.Errorf("dense: SyncToMLX needs FormatNone")
	}
	dt := nativemlx.DTypeName(l.Weights.DType.String())
	if err := nativemlx.RequireLayer("dense", dt); err != nil {
		return err
	}
	l.mlxPrepOK = false
	return l.ensureMLXMirror().CaptureDense(l.Weights, l.Weights.Bias, dt)
}

// SyncFromMLX writes the mirrored snapshot back into Go Weights (API symmetry).
func (l *Layer) SyncFromMLX() error {
	if l == nil || l.mlxMir == nil {
		return nil
	}
	return l.mlxMir.SyncToGo()
}

// MLXSynced reports whether SyncToMLX has captured buffers.
func (l *Layer) MLXSynced() bool {
	return l != nil && l.mlxMir != nil && l.mlxMir.Synced()
}

// PrepareMLX bakes W into an MLX graph for this batch/in/out (outside timed loops).
func (l *Layer) PrepareMLX(batch int) error {
	if l == nil || !l.MLXSynced() {
		return fmt.Errorf("dense: SyncToMLX required before PrepareMLX")
	}
	in, out := l.Core.InputHeight, l.Core.OutputHeight
	w, err := l.mlxMir.Buf(0)
	if err != nil {
		return err
	}
	bias, err := l.mlxMir.Bias()
	if err != nil {
		return err
	}
	sess, err := nativemlx.DefaultSession()
	if err != nil {
		return err
	}
	dt := l.mlxMir.DType()
	if err := sess.DensePrepare(int32(batch), int32(in), int32(out), w, bias, dt); err != nil {
		return err
	}
	l.mlxPrepOK = true
	l.mlxPrepB = batch
	l.mlxPrepIn = in
	l.mlxPrepOut = out
	return nil
}

// ForwardMLX runs Dense on Apple MLX GPU. Uses prepared graph when available (execute-only).
func ForwardMLX[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if l == nil || l.Weights == nil || input == nil {
		return nil, nil, fmt.Errorf("dense: ForwardMLX nil")
	}
	if l.mlxMir == nil || !l.mlxMir.Synced() {
		return nil, nil, fmt.Errorf("dense: SyncToMLX required before BackendMLX forward")
	}
	batch, in, out, err := dims(l, input)
	if err != nil {
		return nil, nil, err
	}
	sess, err := nativemlx.DefaultSession()
	if err != nil {
		return nil, nil, err
	}
	xF := core.SliceAsFloat32(input.Data)
	yF := make([]float32, batch*out)

	needPrep := !l.mlxPrepOK || l.mlxPrepB != batch || l.mlxPrepIn != in || l.mlxPrepOut != out
	if needPrep {
		w, err := l.mlxMir.Buf(0)
		if err != nil {
			return nil, nil, err
		}
		bias, err := l.mlxMir.Bias()
		if err != nil {
			return nil, nil, err
		}
		if err := sess.DensePrepare(int32(batch), int32(in), int32(out), w, bias, l.mlxMir.DType()); err != nil {
			return nil, nil, err
		}
		l.mlxPrepOK = true
		l.mlxPrepB, l.mlxPrepIn, l.mlxPrepOut = batch, in, out
	}
	if err := sess.DenseExecuteF32(xF, yF); err != nil {
		return nil, nil, err
	}

	pre = core.NewTensor[T](batch, out)
	post = core.NewTensor[T](batch, out)
	core.SliceFromFloat32(yF, pre.Data)
	if l.Core.Activation == core.ActivationLinear {
		copy(post.Data, pre.Data)
		return pre, post, nil
	}
	for i := 0; i < batch*out; i++ {
		post.Data[i] = core.Activate(pre.Data[i], l.Core.Activation)
	}
	return pre, post, nil
}
