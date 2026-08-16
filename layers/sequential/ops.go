package sequential

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/quant"
)

// Ops is an optional heterogeneous child chain. When non-empty it is the
// forward/backward tape; Children stays the Dense-only constructor path.
// Sequential cannot nest Parallel/Stack (import cycle) — use parallel.Stack.
func (l *Layer) ChildOps() []any {
	if l == nil {
		return nil
	}
	if len(l.Ops) > 0 {
		return l.Ops
	}
	out := make([]any, len(l.Children))
	for i, ch := range l.Children {
		out[i] = ch
	}
	return out
}

// NewFromOps builds Sequential over mixed child Ops (Dense, SwiGLU, RMSNorm, LayerNorm).
func NewFromOps(cfg Config, ops []any) (*Layer, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if len(ops) == 0 {
		return nil, fmt.Errorf("sequential: NewFromOps needs ≥1 child")
	}
	for i, op := range ops {
		if op == nil {
			return nil, fmt.Errorf("sequential: nil child at %d", i)
		}
		if !supportedChild(op) {
			return nil, fmt.Errorf("sequential: unsupported child %T at %d (no Parallel — use Stack)", op, i)
		}
	}
	cfg.Depth = len(ops)
	l := &Layer{
		Core: core.Layer{
			Type:         core.LayerSequential,
			DType:        core.DTypeFloat32,
			Activation:   core.ActivationLinear,
			InputHeight:  cfg.Dim,
			OutputHeight: cfg.Dim,
			TileSize:     32,
			MultiCore:    true,
		},
		Cfg: cfg,
		Ops: append([]any(nil), ops...),
		Exec: core.ExecConfig{
			Backend:   core.BackendCPUTiled,
			MultiCore: true,
			TileSize:  32,
		},
	}
	l.syncChildExec()
	return l, nil
}

func supportedChild(op any) bool {
	switch op.(type) {
	case *dense.Layer, *swiglu.Layer, *rmsnorm.Layer, *layernorm.Layer:
		return true
	default:
		return false
	}
}

func callFwd[T core.Numeric](op any, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	switch v := op.(type) {
	case *dense.Layer:
		return dense.Forward(v, input)
	case *swiglu.Layer:
		return swiglu.Forward(v, input)
	case *rmsnorm.Layer:
		return rmsnorm.Forward(v, input)
	case *layernorm.Layer:
		return layernorm.Forward(v, input)
	default:
		return nil, nil, fmt.Errorf("sequential: fwd unsupported %T", op)
	}
}

func callBwd[T core.Numeric](op any, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	switch v := op.(type) {
	case *dense.Layer:
		return dense.Backward(v, gradOut, input, pre)
	case *swiglu.Layer:
		return swiglu.Backward(v, gradOut, input, pre)
	case *rmsnorm.Layer:
		return rmsnorm.Backward(v, gradOut, input, pre)
	case *layernorm.Layer:
		return layernorm.Backward(v, gradOut, input, pre)
	default:
		return nil, nil, fmt.Errorf("sequential: bwd unsupported %T", op)
	}
}

func childGradWSize(op any) int {
	switch v := op.(type) {
	case *dense.Layer:
		return v.GradWSize()
	case *swiglu.Layer:
		return v.GradWSize()
	case *rmsnorm.Layer:
		return v.GradWSize()
	case *layernorm.Layer:
		return v.GradWSize()
	default:
		return 0
	}
}

func childApplySGD[T core.Numeric](op any, dW *core.Tensor[T], lr float64) error {
	switch v := op.(type) {
	case *dense.Layer:
		return dense.ApplyGradSGD(v, dW, lr)
	case *swiglu.Layer:
		return swiglu.ApplyGradSGD(v, dW, lr)
	case *rmsnorm.Layer:
		return rmsnorm.ApplyGradSGD(v, dW, lr)
	case *layernorm.Layer:
		return layernorm.ApplyGradSGD(v, dW, lr)
	default:
		return fmt.Errorf("sequential: SGD unsupported %T", op)
	}
}

func childSetDType(op any, dt core.DType) error {
	switch v := op.(type) {
	case *dense.Layer:
		return v.SetDType(dt)
	case *swiglu.Layer:
		return v.SetDType(dt)
	case *rmsnorm.Layer:
		return v.SetDType(dt)
	case *layernorm.Layer:
		return v.SetDType(dt)
	default:
		return fmt.Errorf("sequential: SetDType unsupported %T", op)
	}
}

func childPack(op any, format quant.Format) error {
	switch v := op.(type) {
	case *dense.Layer:
		return v.Pack(format)
	case *swiglu.Layer:
		return v.Pack(format)
	case *rmsnorm.Layer:
		return v.Pack(format)
	case *layernorm.Layer:
		return v.Pack(format)
	default:
		return fmt.Errorf("sequential: Pack unsupported %T", op)
	}
}

func childSetExec(op any, exec core.ExecConfig) {
	switch v := op.(type) {
	case *dense.Layer:
		v.Exec = exec
	case *swiglu.Layer:
		v.Exec = exec
		if v.Gate != nil {
			v.Gate.Exec = exec
		}
		if v.Up != nil {
			v.Up.Exec = exec
		}
		if v.Down != nil {
			v.Down.Exec = exec
		}
	case *rmsnorm.Layer:
		v.Exec = exec
	case *layernorm.Layer:
		v.Exec = exec
	}
}
