// Package training owns optimizers and volumetric train steps.
//
// Layer-agnostic: SGD walks the forward tape and dispatches ApplyGradSGD per
// cell Op (*dense.Layer, *mha.Layer, …). No QAT.
package training

import (
	"fmt"

	"github.com/openfluke/welvet/backward"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/cnn1"
	"github.com/openfluke/welvet/dense"
	"github.com/openfluke/welvet/forward"
	"github.com/openfluke/welvet/layernorm"
	"github.com/openfluke/welvet/mha"
	"github.com/openfluke/welvet/rmsnorm"
	"github.com/openfluke/welvet/swiglu"
)

// Config is shared train hyper-params.
type Config struct {
	LR      float64
	Backend core.Backend // stamped onto grid Exec before Place
}

// DefaultConfig returns a sane SGD setup.
func DefaultConfig() Config {
	return Config{LR: 1e-2, Backend: core.BackendCPUTiled}
}

// MSE mean squared error between pred and target (same shape).
func MSE[T core.Numeric](pred, target *core.Tensor[T]) (float64, error) {
	if pred == nil || target == nil || pred.Len() != target.Len() {
		return 0, fmt.Errorf("training: MSE shape")
	}
	var sum float64
	n := pred.Len()
	for i := 0; i < n; i++ {
		d := core.AsFloat64(pred.Data[i]) - core.AsFloat64(target.Data[i])
		sum += d * d
	}
	return sum / float64(n), nil
}

// MSEGrad returns ∂MSE/∂pred = 2/n (pred − target).
func MSEGrad[T core.Numeric](pred, target *core.Tensor[T]) (*core.Tensor[T], error) {
	if pred == nil || target == nil || pred.Len() != target.Len() {
		return nil, fmt.Errorf("training: MSEGrad shape")
	}
	out := core.NewTensor[T](pred.Shape...)
	n := float64(pred.Len())
	scale := 2.0 / n
	for i := 0; i < pred.Len(); i++ {
		d := core.AsFloat64(pred.Data[i]) - core.AsFloat64(target.Data[i])
		out.Data[i] = core.FromFloat64[T](scale * d)
	}
	return out, nil
}

// GradApplier is implemented by layer packages (Dense today).
type GradApplier[T core.Numeric] interface {
	ApplyGradSGD(dW *core.Tensor[T], lr float64) error
}

// denseApplier adapts *dense.Layer to GradApplier without forcing every layer
// into this package's API yet — dispatch uses type switch + this adapter pattern.
type denseApplier[T core.Numeric] struct{ L *dense.Layer }

func (a denseApplier[T]) ApplyGradSGD(dW *core.Tensor[T], lr float64) error {
	return dense.ApplyGradSGD(a.L, dW, lr)
}

// SGD applies one optimizer step from a forward tape + backward grads.
func SGD[T core.Numeric](fwd *forward.Result[T], bwd *backward.Result[T], lr float64) error {
	if fwd == nil || bwd == nil {
		return fmt.Errorf("training: SGD nil tape")
	}
	if lr == 0 {
		return fmt.Errorf("training: LR is 0")
	}
	dw := make(map[[4]int]*core.Tensor[T], len(bwd.GradWs))
	for _, g := range bwd.GradWs {
		dw[[4]int{g.Coord.Z, g.Coord.Y, g.Coord.X, g.Coord.L}] = g.DW
	}
	for _, st := range fwd.Steps {
		key := [4]int{st.Coord.Z, st.Coord.Y, st.Coord.X, st.Coord.L}
		g, ok := dw[key]
		if !ok || g == nil {
			continue
		}
		if err := applyCell[T](st.Cell.Op, g, lr); err != nil {
			return fmt.Errorf("training SGD %v: %w", st.Coord, err)
		}
	}
	return nil
}

func applyCell[T core.Numeric](op any, dW *core.Tensor[T], lr float64) error {
	switch v := op.(type) {
	case *dense.Layer:
		return dense.ApplyGradSGD(v, dW, lr)
	case *mha.Layer:
		return mha.ApplyGradSGD(v, dW, lr)
	case *swiglu.Layer:
		return swiglu.ApplyGradSGD(v, dW, lr)
	case *rmsnorm.Layer:
		return rmsnorm.ApplyGradSGD(v, dW, lr)
	case *layernorm.Layer:
		return layernorm.ApplyGradSGD(v, dW, lr)
	case *cnn1.Layer:
		return cnn1.ApplyGradSGD(v, dW, lr)
	case GradApplier[T]:
		return v.ApplyGradSGD(dW, lr)
	default:
		return fmt.Errorf("no SGD handler for %T (wire ApplyGradSGD when adding the layer)", op)
	}
}

// Step is one full train iteration: forward → MSE → backward → SGD.
func Step[T core.Numeric](fwd *forward.Result[T], target *core.Tensor[T], lr float64) (loss float64, err error) {
	if fwd == nil || fwd.Output == nil {
		return 0, fmt.Errorf("training: Step nil forward")
	}
	loss, err = MSE(fwd.Output, target)
	if err != nil {
		return 0, err
	}
	gy, err := MSEGrad(fwd.Output, target)
	if err != nil {
		return 0, err
	}
	bwd, err := backward.Backward(fwd, gy)
	if err != nil {
		return 0, err
	}
	if err := SGD(fwd, bwd, lr); err != nil {
		return 0, err
	}
	return loss, nil
}
