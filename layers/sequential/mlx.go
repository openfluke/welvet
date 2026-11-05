package sequential

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/softmax"
	"github.com/openfluke/welvet/nativemlx"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/weights"
)

// SyncToMLX rebuilds a welvet.mlx Sequential from children and syncs each
// supported child's weights. Rejects any child the MLX caps / C ABI cannot mount.
//
// After SyncToMLX, set Exec.Backend = core.BackendMLX for a host-ping-pong Sequential
// forward, or leave BackendCPUTiled and set individual children to BackendMLX.
func (l *Layer) SyncToMLX() error {
	if l == nil {
		return fmt.Errorf("sequential: SyncToMLX nil")
	}
	if !nativemlx.Available() {
		return fmt.Errorf("sequential: SyncToMLX needs libwelvet_mlx")
	}
	ops := l.ChildOps()
	if len(ops) == 0 {
		return fmt.Errorf("sequential: SyncToMLX empty")
	}
	seq, err := nativemlx.NewSequential()
	if err != nil {
		return err
	}
	for i, op := range ops {
		if err := pushMLXChild(seq, op); err != nil {
			seq.Free()
			return fmt.Errorf("sequential: SyncToMLX child %d: %w", i, err)
		}
		if err := syncChildToMLX(op); err != nil {
			seq.Free()
			return fmt.Errorf("sequential: SyncToMLX child %d sync: %w", i, err)
		}
	}
	if l.mlxSeq != nil {
		l.mlxSeq.Free()
	}
	l.mlxSeq = seq
	l.mlxSynced = true
	return nil
}

func pushMLXChild(seq *nativemlx.Sequential, op any) error {
	switch v := op.(type) {
	case *dense.Layer:
		if v.Weights == nil || v.Weights.Format != quant.FormatNone {
			return fmt.Errorf("dense needs FormatNone")
		}
		// Sequential C ABI push is float32-only today.
		if err := nativemlx.RequireLayer("dense", "float32"); err != nil {
			return err
		}
		w, err := weights.MatrixF32(v.Weights)
		if err != nil {
			return err
		}
		bias := biasF32(v.Weights.Bias)
		return seq.PushDense(int32(v.Core.InputHeight), int32(v.Core.OutputHeight), w, bias)
	case *rmsnorm.Layer:
		if v.Gamma == nil || v.Gamma.Format != quant.FormatNone {
			return fmt.Errorf("rmsnorm needs FormatNone γ")
		}
		if err := nativemlx.RequireLayer("rmsnorm", "float32"); err != nil {
			return err
		}
		_ = v.Cfg.Validate()
		g, err := v.Gamma.FlattenF32()
		if err != nil {
			return err
		}
		return seq.PushRmsNorm(int32(v.Cfg.Dim), float32(v.Cfg.Eps), g)
	case *layernorm.Layer:
		if v.Gamma == nil || v.Beta == nil {
			return fmt.Errorf("layernorm nil γ/β")
		}
		if v.Gamma.Format != quant.FormatNone || v.Beta.Format != quant.FormatNone {
			return fmt.Errorf("layernorm needs FormatNone")
		}
		if err := nativemlx.RequireLayer("layernorm", "float32"); err != nil {
			return err
		}
		_ = v.Cfg.Validate()
		g, err := v.Gamma.FlattenF32()
		if err != nil {
			return err
		}
		b, err := v.Beta.FlattenF32()
		if err != nil {
			return err
		}
		return seq.PushLayerNorm(int32(v.Cfg.Dim), float32(v.Cfg.Eps), g, b)
	case *softmax.Layer:
		if err := nativemlx.RequireLayer("softmax", "float32"); err != nil {
			return err
		}
		return seq.PushSoftmax(int32(v.Cfg.Dim))
	default:
		return fmt.Errorf("unsupported MLX mount %T (caps/C ABI)", op)
	}
}

func syncChildToMLX(op any) error {
	switch v := op.(type) {
	case *dense.Layer:
		return v.SyncToMLX()
	case *rmsnorm.Layer:
		return v.SyncToMLX()
	case *layernorm.Layer:
		return v.SyncToMLX()
	case *softmax.Layer:
		return v.SyncToMLX()
	default:
		return fmt.Errorf("no SyncToMLX on %T", op)
	}
}

// SyncFromMLX pulls child mirrors back into Go Stores.
func (l *Layer) SyncFromMLX() error {
	if l == nil {
		return nil
	}
	for i, op := range l.ChildOps() {
		if err := syncChildFromMLX(op); err != nil {
			return fmt.Errorf("sequential: SyncFromMLX child %d: %w", i, err)
		}
	}
	return nil
}

func syncChildFromMLX(op any) error {
	switch v := op.(type) {
	case *dense.Layer:
		return v.SyncFromMLX()
	case *rmsnorm.Layer:
		return v.SyncFromMLX()
	case *layernorm.Layer:
		return v.SyncFromMLX()
	case *softmax.Layer:
		return v.SyncFromMLX()
	default:
		return nil
	}
}

// MLXSynced reports whether SyncToMLX built a native Sequential.
func (l *Layer) MLXSynced() bool {
	return l != nil && l.mlxSynced && l.mlxSeq != nil
}

// ForwardMLX runs the welvet.mlx Sequential on Metal GPU (host ping-pong).
func ForwardMLX[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if l == nil || input == nil {
		return nil, nil, fmt.Errorf("sequential: ForwardMLX nil")
	}
	if !l.MLXSynced() {
		return nil, nil, fmt.Errorf("sequential: SyncToMLX required before BackendMLX forward")
	}
	lay, err := parseLayout(l.Cfg.Dim, input)
	if err != nil {
		return nil, nil, err
	}
	flat := flatten(input, lay)
	dim := l.Cfg.Dim
	sess, err := nativemlx.DefaultSession()
	if err != nil {
		return nil, nil, err
	}
	xF := core.SliceAsFloat32(flat.Data)
	yF := make([]float32, len(xF))
	if err := l.mlxSeq.ForwardF32(sess, int32(lay.batch), int32(lay.seq), xF, yF); err != nil {
		return nil, nil, err
	}
	outFlat := core.NewTensor[T](flat.Shape...)
	core.SliceFromFloat32(yF, outFlat.Data)
	pre = unflatten(outFlat, lay, dim)
	post = unflatten(outFlat, lay, dim)
	return pre, post, nil
}
