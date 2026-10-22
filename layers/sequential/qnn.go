package sequential

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/nativeqnn"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/weights"
)

// SyncToQNN rebuilds a welvet.qnn Sequential from children and syncs each
// supported child's weights. Rejects any child the HTP caps / C ABI cannot mount
// (e.g. SwiGLU until push_swiglu exists).
//
// After SyncToQNN, set Exec.Backend = core.BackendQNN for a fused NPU forward,
// or leave BackendCPUTiled and set individual children to BackendQNN for mixed Go/NPU.
func (l *Layer) SyncToQNN() error {
	if l == nil {
		return fmt.Errorf("sequential: SyncToQNN nil")
	}
	if !nativeqnn.Available() {
		return fmt.Errorf("sequential: SyncToQNN needs libwelvet_qnn")
	}
	ops := l.ChildOps()
	if len(ops) == 0 {
		return fmt.Errorf("sequential: SyncToQNN empty")
	}
	seq, err := nativeqnn.NewSequential()
	if err != nil {
		return err
	}
	for i, op := range ops {
		if err := pushQNNChild(seq, op); err != nil {
			seq.Free()
			return fmt.Errorf("sequential: SyncToQNN child %d: %w", i, err)
		}
		if err := syncChildToQNN(op); err != nil {
			seq.Free()
			return fmt.Errorf("sequential: SyncToQNN child %d sync: %w", i, err)
		}
	}
	if l.qnnSeq != nil {
		l.qnnSeq.Free()
	}
	l.qnnSeq = seq
	l.qnnSynced = true
	return nil
}

func pushQNNChild(seq *nativeqnn.Sequential, op any) error {
	switch v := op.(type) {
	case *dense.Layer:
		if v.Weights == nil || v.Weights.Format != quant.FormatNone {
			return fmt.Errorf("dense needs FormatNone")
		}
		if err := nativeqnn.RequireLayer("dense", "float32"); err != nil {
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
		if err := nativeqnn.RequireLayer("rmsnorm", "float32"); err != nil {
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
		if err := nativeqnn.RequireLayer("layernorm", "float32"); err != nil {
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
	default:
		return fmt.Errorf("unsupported QNN mount %T (caps/C ABI)", op)
	}
}

func syncChildToQNN(op any) error {
	switch v := op.(type) {
	case *dense.Layer:
		return v.SyncToQNN()
	case *rmsnorm.Layer:
		return v.SyncToQNN()
	case *layernorm.Layer:
		return v.SyncToQNN()
	default:
		return fmt.Errorf("no SyncToQNN on %T", op)
	}
}

func biasF32(b []float64) []float32 {
	if len(b) == 0 {
		return nil
	}
	out := make([]float32, len(b))
	for i, v := range b {
		out[i] = float32(v)
	}
	return out
}

// SyncFromQNN pulls child mirrors back into Go Stores.
func (l *Layer) SyncFromQNN() error {
	if l == nil {
		return nil
	}
	for i, op := range l.ChildOps() {
		if err := syncChildFromQNN(op); err != nil {
			return fmt.Errorf("sequential: SyncFromQNN child %d: %w", i, err)
		}
	}
	return nil
}

func syncChildFromQNN(op any) error {
	switch v := op.(type) {
	case *dense.Layer:
		return v.SyncFromQNN()
	case *rmsnorm.Layer:
		return v.SyncFromQNN()
	case *layernorm.Layer:
		return v.SyncFromQNN()
	default:
		return nil
	}
}

// QNNSynced reports whether SyncToQNN built a native Sequential.
func (l *Layer) QNNSynced() bool {
	return l != nil && l.qnnSynced && l.qnnSeq != nil
}

// ForwardQNN runs the fused welvet.qnn Sequential on HTP.
func ForwardQNN[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if l == nil || input == nil {
		return nil, nil, fmt.Errorf("sequential: ForwardQNN nil")
	}
	if !l.QNNSynced() {
		return nil, nil, fmt.Errorf("sequential: SyncToQNN required before BackendQNN forward")
	}
	lay, err := parseLayout(l.Cfg.Dim, input)
	if err != nil {
		return nil, nil, err
	}
	flat := flatten(input, lay)
	dim := l.Cfg.Dim
	sess, err := nativeqnn.DefaultSession()
	if err != nil {
		return nil, nil, err
	}
	xF := core.SliceAsFloat32(flat.Data)
	yF := make([]float32, len(xF))
	// C ABI: batch × seq_len tokens of in_features
	if err := l.qnnSeq.ForwardF32(sess, int32(lay.batch), int32(lay.seq), xF, yF); err != nil {
		return nil, nil, err
	}
	outFlat := core.NewTensor[T](flat.Shape...)
	core.SliceFromFloat32(yF, outFlat.Data)
	pre = unflatten(outFlat, lay, dim)
	post = unflatten(outFlat, lay, dim)
	return pre, post, nil
}
