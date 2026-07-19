// Package training owns optimizers and volumetric train steps.
//
// Layer-agnostic: SGD walks the forward tape and dispatches ApplyGradSGD per
// cell Op (*dense.Layer, *mha.Layer, …). No QAT.
package training

import (
	"fmt"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/backward"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/cnn2"
	"github.com/openfluke/welvet/layers/cnn3"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/embedding"
	"github.com/openfluke/welvet/forward"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/lstm"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/rnn"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/layers/softmax"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/step"
	"github.com/openfluke/welvet/tween"
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
	case *cnn2.Layer:
		return cnn2.ApplyGradSGD(v, dW, lr)
	case *cnn3.Layer:
		return cnn3.ApplyGradSGD(v, dW, lr)
	case *rnn.Layer:
		return rnn.ApplyGradSGD(v, dW, lr)
	case *lstm.Layer:
		return lstm.ApplyGradSGD(v, dW, lr)
	case *embedding.Layer:
		return embedding.ApplyGradSGD(v, dW, lr)
	case *softmax.Layer:
		return softmax.ApplyGradSGD(v, dW, lr)
	case *sequential.Layer:
		return sequential.ApplyGradSGD(v, dW, lr)
	case *residual.Layer:
		return residual.ApplyGradSGD(v, dW, lr)
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

// ApplyTween runs one target-propagation step from a forward tape + target.
// Uses chain-rule tween by default (Config.UseChainRule).
func ApplyTween[T core.Numeric](g *architecture.Grid, fwd *forward.Result[T], input, target *core.Tensor[T], lr float64) (*tween.State[T], error) {
	if g == nil || fwd == nil || target == nil {
		return nil, fmt.Errorf("training: ApplyTween nil args")
	}
	cfg := tween.DefaultConfig()
	cfg.LearningRate = float32(lr)
	st := tween.NewState[T](g, cfg)
	tween.CaptureFromForward(st, fwd, input)
	if err := tween.Backward(g, st, target); err != nil {
		return nil, err
	}
	st.CalculateLinkBudgets()
	if err := tween.ApplyGaps(g, st, float32(lr)); err != nil {
		return nil, err
	}
	return st, nil
}

// StepTween is forward → MSE → tween gap update (alternative to Step/SGD).
func StepTween[T core.Numeric](g *architecture.Grid, input, target *core.Tensor[T], lr float64) (loss float64, st *tween.State[T], err error) {
	fwd, err := forward.Forward(g, input)
	if err != nil {
		return 0, nil, err
	}
	loss, err = MSE(fwd.Output, target)
	if err != nil {
		return 0, nil, err
	}
	st, err = ApplyTween(g, fwd, input, target, lr)
	return loss, st, err
}

// StepMesh runs ticks of volumetric step.Forward then ApplyTween (gap-based).
// Honors g.Exec / per-Op Backend (CPU tiled / SIMD).
func StepMesh[T core.Numeric](g *architecture.Grid, input, target *core.Tensor[T], ticks int, lr float64) (loss float64, st *step.State[T], err error) {
	if g == nil || input == nil || target == nil {
		return 0, nil, fmt.Errorf("training: StepMesh nil args")
	}
	if ticks < 1 {
		ticks = 1
	}
	st = step.New[T](g)
	st.SetInput(input)
	for t := 0; t < ticks; t++ {
		capture := t == ticks-1
		if _, err := step.Forward(g, st, capture); err != nil {
			return 0, st, err
		}
	}
	out := st.LayerData[len(st.LayerData)-1]
	if out == nil {
		return 0, st, fmt.Errorf("training: StepMesh nil mesh output")
	}
	loss, err = MSE(out, target)
	if err != nil {
		return loss, st, err
	}
	if err := step.ApplyTween(g, st, target, float32(lr)); err != nil {
		return loss, st, err
	}
	return loss, st, nil
}
