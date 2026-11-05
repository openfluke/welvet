package rmsnorm

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

// SyncToMLX pushes γ into the MLX mirror (caps-checked).
func (l *Layer) SyncToMLX() error {
	if l == nil || l.Gamma == nil {
		return fmt.Errorf("rmsnorm: SyncToMLX nil")
	}
	if l.Gamma.Format != quant.FormatNone {
		return fmt.Errorf("rmsnorm: SyncToMLX needs FormatNone")
	}
	_ = l.Cfg.Validate()
	dt := nativemlx.DTypeName(l.Gamma.DType.String())
	return l.ensureMLXMirror().CaptureNorm("rmsnorm", float32(l.Cfg.Eps), dt, l.Gamma, nil)
}

// SyncFromMLX writes mirrored γ back to Go.
func (l *Layer) SyncFromMLX() error {
	if l == nil || l.mlxMir == nil {
		return nil
	}
	return l.mlxMir.SyncToGo()
}

// MLXSynced reports whether SyncToMLX captured γ.
func (l *Layer) MLXSynced() bool {
	return l != nil && l.mlxMir != nil && l.mlxMir.Synced()
}

// ForwardMLX runs RMSNorm on Apple MLX GPU.
func ForwardMLX[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if l == nil || l.Gamma == nil || input == nil {
		return nil, nil, fmt.Errorf("rmsnorm: ForwardMLX nil")
	}
	if l.mlxMir == nil || !l.mlxMir.Synced() {
		return nil, nil, fmt.Errorf("rmsnorm: SyncToMLX required before BackendMLX forward")
	}
	gamma, err := l.mlxMir.Buf(0)
	if err != nil {
		return nil, nil, err
	}
	eps, err := l.mlxMir.Eps()
	if err != nil {
		return nil, nil, err
	}
	batch, seq, dim, err := rmsDims(l, input)
	if err != nil {
		return nil, nil, err
	}
	sess, err := nativemlx.DefaultSession()
	if err != nil {
		return nil, nil, err
	}
	xF := core.SliceAsFloat32(input.Data)
	yF := make([]float32, len(xF))
	if err := sess.RmsNormForward(int32(batch), int32(seq), int32(dim), eps, gamma, xF, yF, l.mlxMir.DType()); err != nil {
		return nil, nil, err
	}
	out := core.NewTensor[T](input.Shape...)
	core.SliceFromFloat32(yF, out.Data)
	return out, out, nil
}
