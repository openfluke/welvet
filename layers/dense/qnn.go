package dense

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/nativeqnn"
	"github.com/openfluke/welvet/quant"
)

func (l *Layer) ensureQNNMirror() *nativeqnn.WeightMirror {
	if l.qnnMir == nil {
		l.qnnMir = &nativeqnn.WeightMirror{}
	}
	return l.qnnMir
}

// SyncToQNN copies Go Weights (+ Bias) into the QNN mirror and checks HTP caps.
// Invalidates any prepared HTP graph — next ForwardQNN will prepare then execute.
func (l *Layer) SyncToQNN() error {
	if l == nil || l.Weights == nil {
		return fmt.Errorf("dense: SyncToQNN nil")
	}
	if l.Weights.Format != quant.FormatNone {
		return fmt.Errorf("dense: SyncToQNN needs FormatNone")
	}
	dt := nativeqnn.DTypeName(l.Weights.DType.String())
	if err := nativeqnn.RequireLayer("dense", dt); err != nil {
		return err
	}
	if dt != "float32" {
		return fmt.Errorf("dense: SyncToQNN C ABI is float32 today (got %s)", dt)
	}
	l.qnnPrepOK = false
	return l.ensureQNNMirror().CaptureDense(l.Weights, l.Weights.Bias)
}

// SyncFromQNN writes the mirrored snapshot back into Go Weights (API symmetry).
func (l *Layer) SyncFromQNN() error {
	if l == nil || l.qnnMir == nil {
		return nil
	}
	return l.qnnMir.SyncToGo()
}

// QNNSynced reports whether SyncToQNN has captured buffers.
func (l *Layer) QNNSynced() bool {
	return l != nil && l.qnnMir != nil && l.qnnMir.Synced()
}

// PrepareQNN bakes W into an HTP graph for this batch/in/out (outside timed loops).
func (l *Layer) PrepareQNN(batch int) error {
	if l == nil || !l.QNNSynced() {
		return fmt.Errorf("dense: SyncToQNN required before PrepareQNN")
	}
	in, out := l.Core.InputHeight, l.Core.OutputHeight
	w, err := l.qnnMir.Buf(0)
	if err != nil {
		return err
	}
	bias, err := l.qnnMir.Bias()
	if err != nil {
		return err
	}
	sess, err := nativeqnn.DefaultSession()
	if err != nil {
		return err
	}
	if err := sess.DensePrepareF32(int32(batch), int32(in), int32(out), w, bias); err != nil {
		return err
	}
	l.qnnPrepOK = true
	l.qnnPrepB = batch
	l.qnnPrepIn = in
	l.qnnPrepOut = out
	return nil
}

// ForwardQNN runs Dense on Hexagon HTP. Uses prepared graph when available (execute-only).
func ForwardQNN[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if l == nil || l.Weights == nil || input == nil {
		return nil, nil, fmt.Errorf("dense: ForwardQNN nil")
	}
	if l.qnnMir == nil || !l.qnnMir.Synced() {
		return nil, nil, fmt.Errorf("dense: SyncToQNN required before BackendQNN forward")
	}
	batch, in, out, err := dims(l, input)
	if err != nil {
		return nil, nil, err
	}
	sess, err := nativeqnn.DefaultSession()
	if err != nil {
		return nil, nil, err
	}
	xF := core.SliceAsFloat32(input.Data)
	yF := make([]float32, batch*out)

	needPrep := !l.qnnPrepOK || l.qnnPrepB != batch || l.qnnPrepIn != in || l.qnnPrepOut != out
	if needPrep {
		w, err := l.qnnMir.Buf(0)
		if err != nil {
			return nil, nil, err
		}
		bias, err := l.qnnMir.Bias()
		if err != nil {
			return nil, nil, err
		}
		if err := sess.DensePrepareF32(int32(batch), int32(in), int32(out), w, bias); err != nil {
			return nil, nil, err
		}
		l.qnnPrepOK = true
		l.qnnPrepB, l.qnnPrepIn, l.qnnPrepOut = batch, in, out
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
