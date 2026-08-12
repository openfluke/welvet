package parallel

import (
	"fmt"

	"github.com/openfluke/welvet/core"
)

// TrainStackMSE runs ForwardStack → MSE → TrainStack under mode.
// Covers SGD / Tween / TweenChain (and per-child overrides via ChildModes).
func TrainStackMSE[T core.Numeric](s *Stack, input, target *core.Tensor[T], mode TrainMode, lr float64) (loss float64, err error) {
	if s == nil || input == nil || target == nil {
		return 0, fmt.Errorf("parallel: TrainStackMSE nil")
	}
	pre, post, err := ForwardStack(s, input)
	if err != nil {
		return 0, err
	}
	loss, gy, err := mseGrad(post, target)
	if err != nil {
		return 0, err
	}
	if err := TrainStack(s, gy, input, pre, mode, lr); err != nil {
		return loss, err
	}
	return loss, nil
}

// TrainMSE runs Forward → MSE → Train under mode (Parallel root).
func TrainMSE[T core.Numeric](l *Layer, input, target *core.Tensor[T], mode TrainMode, lr float64) (loss float64, err error) {
	if l == nil || input == nil || target == nil {
		return 0, fmt.Errorf("parallel: TrainMSE nil")
	}
	pre, post, err := Forward(l, input)
	if err != nil {
		return 0, err
	}
	loss, gy, err := mseGrad(post, target)
	if err != nil {
		return 0, err
	}
	if err := Train(l, gy, input, pre, mode, lr); err != nil {
		return loss, err
	}
	return loss, nil
}

func mseGrad[T core.Numeric](pred, target *core.Tensor[T]) (loss float64, gy *core.Tensor[T], err error) {
	if pred == nil || target == nil || pred.Len() != target.Len() {
		return 0, nil, fmt.Errorf("parallel: mse shape mismatch")
	}
	gy = core.NewTensor[T](pred.Shape...)
	var sum float64
	n := pred.Len()
	for i := 0; i < n; i++ {
		d := core.AsFloat64(pred.Data[i]) - core.AsFloat64(target.Data[i])
		sum += d * d
		gy.Data[i] = core.FromFloat64[T](2 * d / float64(n))
	}
	return sum / float64(n), gy, nil
}
