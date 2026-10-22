package rmsnorm

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

// SyncToQNN pushes γ into the QNN mirror (caps-checked).
func (l *Layer) SyncToQNN() error {
	if l == nil || l.Gamma == nil {
		return fmt.Errorf("rmsnorm: SyncToQNN nil")
	}
	if l.Gamma.Format != quant.FormatNone {
		return fmt.Errorf("rmsnorm: SyncToQNN needs FormatNone")
	}
	_ = l.Cfg.Validate()
	return l.ensureQNNMirror().CaptureNorm("rmsnorm", float32(l.Cfg.Eps), l.Gamma, nil)
}

// SyncFromQNN writes mirrored γ back to Go.
func (l *Layer) SyncFromQNN() error {
	if l == nil || l.qnnMir == nil {
		return nil
	}
	return l.qnnMir.SyncToGo()
}

// QNNSynced reports whether SyncToQNN captured γ.
func (l *Layer) QNNSynced() bool {
	return l != nil && l.qnnMir != nil && l.qnnMir.Synced()
}

// ForwardQNN runs RMSNorm on Hexagon HTP.
func ForwardQNN[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if l == nil || l.Gamma == nil || input == nil {
		return nil, nil, fmt.Errorf("rmsnorm: ForwardQNN nil")
	}
	if l.qnnMir == nil || !l.qnnMir.Synced() {
		return nil, nil, fmt.Errorf("rmsnorm: SyncToQNN required before BackendQNN forward")
	}
	gamma, err := l.qnnMir.Buf(0)
	if err != nil {
		return nil, nil, err
	}
	eps, err := l.qnnMir.Eps()
	if err != nil {
		return nil, nil, err
	}
	batch, seq, dim, err := rmsDims(l, input)
	if err != nil {
		return nil, nil, err
	}
	sess, err := nativeqnn.DefaultSession()
	if err != nil {
		return nil, nil, err
	}
	xF := core.SliceAsFloat32(input.Data)
	yF := make([]float32, len(xF))
	if err := sess.RmsNormForwardF32(int32(batch), int32(seq), int32(dim), eps, gamma, xF, yF); err != nil {
		return nil, nil, err
	}
	out := core.NewTensor[T](input.Shape...)
	core.SliceFromFloat32(yF, out.Data)
	return out, out, nil
}

func rmsDims[T core.Numeric](l *Layer, input *core.Tensor[T]) (batch, seq, dim int, err error) {
	dim = l.Cfg.Dim
	switch len(input.Shape) {
	case 2:
		return input.Shape[0], 1, dim, checkLast(input.Shape[1], dim)
	case 3:
		return input.Shape[0], input.Shape[1], dim, checkLast(input.Shape[2], dim)
	default:
		return 0, 0, 0, fmt.Errorf("rmsnorm: shape need [B,D] or [B,S,D]")
	}
}

func checkLast(got, want int) error {
	if got != want {
		return fmt.Errorf("rmsnorm: last dim %d != %d", got, want)
	}
	return nil
}
