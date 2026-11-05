package layernorm

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

// SyncToMLX pushes γ,β into the MLX mirror (caps-checked).
func (l *Layer) SyncToMLX() error {
	if l == nil || l.Gamma == nil || l.Beta == nil {
		return fmt.Errorf("layernorm: SyncToMLX nil")
	}
	if l.Gamma.Format != quant.FormatNone || l.Beta.Format != quant.FormatNone {
		return fmt.Errorf("layernorm: SyncToMLX needs FormatNone")
	}
	_ = l.Cfg.Validate()
	dt := nativemlx.DTypeName(l.Gamma.DType.String())
	return l.ensureMLXMirror().CaptureNorm("layernorm", float32(l.Cfg.Eps), dt, l.Gamma, l.Beta)
}

// SyncFromMLX writes mirrored γ,β back to Go.
func (l *Layer) SyncFromMLX() error {
	if l == nil || l.mlxMir == nil {
		return nil
	}
	return l.mlxMir.SyncToGo()
}

// MLXSynced reports whether SyncToMLX captured params.
func (l *Layer) MLXSynced() bool {
	return l != nil && l.mlxMir != nil && l.mlxMir.Synced()
}

// ForwardMLX runs LayerNorm on Apple MLX GPU.
func ForwardMLX[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if l == nil || l.Gamma == nil || l.Beta == nil || input == nil {
		return nil, nil, fmt.Errorf("layernorm: ForwardMLX nil")
	}
	if l.mlxMir == nil || !l.mlxMir.Synced() {
		return nil, nil, fmt.Errorf("layernorm: SyncToMLX required before BackendMLX forward")
	}
	gamma, err := l.mlxMir.Buf(0)
	if err != nil {
		return nil, nil, err
	}
	beta, err := l.mlxMir.Buf(1)
	if err != nil {
		return nil, nil, err
	}
	eps, err := l.mlxMir.Eps()
	if err != nil {
		return nil, nil, err
	}
	batch, seq, dim, err := lnDims(l, input)
	if err != nil {
		return nil, nil, err
	}
	sess, err := nativemlx.DefaultSession()
	if err != nil {
		return nil, nil, err
	}
	xF := core.SliceAsFloat32(input.Data)
	yF := make([]float32, len(xF))
	if err := sess.LayerNormForward(int32(batch), int32(seq), int32(dim), eps, gamma, beta, xF, yF, l.mlxMir.DType()); err != nil {
		return nil, nil, err
	}
	out := core.NewTensor[T](input.Shape...)
	core.SliceFromFloat32(yF, out.Data)
	return out, out, nil
}
