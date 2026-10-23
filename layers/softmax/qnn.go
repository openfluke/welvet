package softmax

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/nativeqnn"
)

func (l *Layer) ensureQNNMirror() *nativeqnn.WeightMirror {
	if l.qnnMir == nil {
		l.qnnMir = &nativeqnn.WeightMirror{}
	}
	return l.qnnMir
}

// SyncToQNN caps-checks Softmax (weightless) and marks the layer ready for BackendQNN.
func (l *Layer) SyncToQNN() error {
	if l == nil {
		return fmt.Errorf("softmax: SyncToQNN nil")
	}
	return l.ensureQNNMirror().MarkSynced("softmax")
}

// SyncFromQNN is a no-op for weightless Softmax.
func (l *Layer) SyncFromQNN() error { return nil }

// QNNSynced reports whether SyncToQNN succeeded.
func (l *Layer) QNNSynced() bool {
	return l != nil && l.qnnMir != nil && l.qnnMir.Synced()
}

// ForwardQNN runs Softmax on Hexagon HTP.
func ForwardQNN[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if l == nil || input == nil {
		return nil, nil, fmt.Errorf("softmax: ForwardQNN nil")
	}
	if l.qnnMir == nil || !l.qnnMir.Synced() {
		return nil, nil, fmt.Errorf("softmax: SyncToQNN required before BackendQNN forward")
	}
	cols := l.Cfg.Dim
	n := len(input.Data)
	if cols <= 0 || n%cols != 0 {
		return nil, nil, fmt.Errorf("softmax: data len %d not multiple of Dim %d", n, cols)
	}
	rows := n / cols
	sess, err := nativeqnn.DefaultSession()
	if err != nil {
		return nil, nil, err
	}
	xF := core.SliceAsFloat32(input.Data)
	yF := make([]float32, n)
	if err := sess.SoftmaxForwardF32(int32(rows), int32(cols), xF, yF); err != nil {
		return nil, nil, err
	}
	out := core.NewTensor[T](input.Shape...)
	core.SliceFromFloat32(yF, out.Data)
	return out, out, nil
}
