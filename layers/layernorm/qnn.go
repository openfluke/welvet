package layernorm

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

// SyncToQNN pushes γ,β into the QNN mirror (caps-checked).
func (l *Layer) SyncToQNN() error {
	if l == nil || l.Gamma == nil || l.Beta == nil {
		return fmt.Errorf("layernorm: SyncToQNN nil")
	}
	if l.Gamma.Format != quant.FormatNone || l.Beta.Format != quant.FormatNone {
		return fmt.Errorf("layernorm: SyncToQNN needs FormatNone")
	}
	_ = l.Cfg.Validate()
	return l.ensureQNNMirror().CaptureNorm("layernorm", float32(l.Cfg.Eps), l.Gamma, l.Beta)
}

// SyncFromQNN writes mirrored γ,β back to Go.
func (l *Layer) SyncFromQNN() error {
	if l == nil || l.qnnMir == nil {
		return nil
	}
	return l.qnnMir.SyncToGo()
}

// QNNSynced reports whether SyncToQNN captured params.
func (l *Layer) QNNSynced() bool {
	return l != nil && l.qnnMir != nil && l.qnnMir.Synced()
}

// ForwardQNN runs LayerNorm on Hexagon HTP.
func ForwardQNN[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if l == nil || l.Gamma == nil || l.Beta == nil || input == nil {
		return nil, nil, fmt.Errorf("layernorm: ForwardQNN nil")
	}
	if l.qnnMir == nil || !l.qnnMir.Synced() {
		return nil, nil, fmt.Errorf("layernorm: SyncToQNN required before BackendQNN forward")
	}
	gamma, err := l.qnnMir.Buf(0)
	if err != nil {
		return nil, nil, err
	}
	beta, err := l.qnnMir.Buf(1)
	if err != nil {
		return nil, nil, err
	}
	eps, err := l.qnnMir.Eps()
	if err != nil {
		return nil, nil, err
	}
	batch, seq, dim, err := lnDims(l, input)
	if err != nil {
		return nil, nil, err
	}
	sess, err := nativeqnn.DefaultSession()
	if err != nil {
		return nil, nil, err
	}
	xF := core.SliceAsFloat32(input.Data)
	yF := make([]float32, len(xF))
	if err := sess.LayerNormForwardF32(int32(batch), int32(seq), int32(dim), eps, gamma, beta, xF, yF); err != nil {
		return nil, nil, err
	}
	out := core.NewTensor[T](input.Shape...)
	core.SliceFromFloat32(yF, out.Data)
	return out, out, nil
}

func lnDims[T core.Numeric](l *Layer, input *core.Tensor[T]) (batch, seq, dim int, err error) {
	dim = l.Cfg.Dim
	switch len(input.Shape) {
	case 2:
		if input.Shape[1] != dim {
			return 0, 0, 0, fmt.Errorf("layernorm: last dim %d != %d", input.Shape[1], dim)
		}
		return input.Shape[0], 1, dim, nil
	case 3:
		if input.Shape[2] != dim {
			return 0, 0, 0, fmt.Errorf("layernorm: last dim %d != %d", input.Shape[2], dim)
		}
		return input.Shape[0], input.Shape[1], dim, nil
	default:
		return 0, 0, 0, fmt.Errorf("layernorm: shape need [B,D] or [B,S,D]")
	}
}
