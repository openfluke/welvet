package softmax

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/nativemlx"
)

func (l *Layer) ensureMLXMirror() *nativemlx.WeightMirror {
	if l.mlxMir == nil {
		l.mlxMir = &nativemlx.WeightMirror{}
	}
	return l.mlxMir
}

// SyncToMLX caps-checks Softmax (weightless) and marks the layer ready for BackendMLX.
func (l *Layer) SyncToMLX() error {
	if l == nil {
		return fmt.Errorf("softmax: SyncToMLX nil")
	}
	dt := nativemlx.DTypeName(l.Core.DType.String())
	return l.ensureMLXMirror().MarkSynced("softmax", dt)
}

// SyncFromMLX is a no-op for weightless Softmax.
func (l *Layer) SyncFromMLX() error { return nil }

// MLXSynced reports whether SyncToMLX succeeded.
func (l *Layer) MLXSynced() bool {
	return l != nil && l.mlxMir != nil && l.mlxMir.Synced()
}

// ForwardMLX runs Softmax on Apple MLX GPU.
func ForwardMLX[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if l == nil || input == nil {
		return nil, nil, fmt.Errorf("softmax: ForwardMLX nil")
	}
	if l.mlxMir == nil || !l.mlxMir.Synced() {
		return nil, nil, fmt.Errorf("softmax: SyncToMLX required before BackendMLX forward")
	}
	cols := l.Cfg.Dim
	n := len(input.Data)
	if cols <= 0 || n%cols != 0 {
		return nil, nil, fmt.Errorf("softmax: data len %d not multiple of Dim %d", n, cols)
	}
	rows := n / cols
	sess, err := nativemlx.DefaultSession()
	if err != nil {
		return nil, nil, err
	}
	xF := core.SliceAsFloat32(input.Data)
	yF := make([]float32, n)
	if err := sess.SoftmaxForward(int32(rows), int32(cols), xF, yF, l.mlxMir.DType()); err != nil {
		return nil, nil, err
	}
	out := core.NewTensor[T](input.Shape...)
	core.SliceFromFloat32(yF, out.Data)
	return out, out, nil
}
