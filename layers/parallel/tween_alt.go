package parallel

import (
	"fmt"

	"github.com/openfluke/welvet/core"
)

func altTimesOf(n int) int {
	if n < 1 {
		return 1
	}
	return n
}

// trainTweenAltStackMSE: for AltTimes cycles, recompute the MSE gap then
// TweenSplit, recompute, then Tween. Split → Tween → Split → Tween …
func trainTweenAltStackMSE[T core.Numeric](s *Stack, input, target *core.Tensor[T], lr float64) (float64, error) {
	return trainTweenAltStackGap(s, input, target, lr, mseGrad[T])
}

func trainTweenAltStackCE[T core.Numeric](s *Stack, input, target *core.Tensor[T], lr float64) (float64, error) {
	return trainTweenAltStackGap(s, input, target, lr, ceGrad[T])
}

func trainTweenAltStackGap[T core.Numeric](s *Stack, input, target *core.Tensor[T], lr float64, gap outputGap[T]) (float64, error) {
	times := altTimesOf(s.AltTimes)
	var last float64
	for i := 0; i < times; i++ {
		_, post, err := ForwardStack(s, input)
		if err != nil {
			return last, err
		}
		loss, gy, err := gap(post, target)
		if err != nil {
			return last, err
		}
		last = loss
		if err := trainTweenSplitOp(s, gy, input, lr); err != nil {
			return last, fmt.Errorf("tween-alt split %d: %w", i, err)
		}
		_, post, err = ForwardStack(s, input)
		if err != nil {
			return last, err
		}
		loss, gy, err = gap(post, target)
		if err != nil {
			return last, err
		}
		last = loss
		if err := applyProjectedTween(s, gy, input, lr*0.5); err != nil {
			return last, fmt.Errorf("tween-alt tween %d: %w", i, err)
		}
	}
	return last, nil
}

func trainTweenAltLayerMSE[T core.Numeric](l *Layer, input, target *core.Tensor[T], lr float64) (float64, error) {
	times := altTimesOf(l.AltTimes)
	var last float64
	for i := 0; i < times; i++ {
		_, post, err := Forward(l, input)
		if err != nil {
			return last, err
		}
		loss, gy, err := mseGrad(post, target)
		if err != nil {
			return last, err
		}
		last = loss
		if err := trainTweenSplitOp(l, gy, input, lr); err != nil {
			return last, fmt.Errorf("tween-alt split %d: %w", i, err)
		}
		_, post, err = Forward(l, input)
		if err != nil {
			return last, err
		}
		loss, gy, err = mseGrad(post, target)
		if err != nil {
			return last, err
		}
		last = loss
		if err := applyProjectedTween(l, gy, input, lr*0.5); err != nil {
			return last, fmt.Errorf("tween-alt tween %d: %w", i, err)
		}
	}
	return last, nil
}

// trainTweenAltOp: same Split→Tween loop using a frozen gap (no target).
func trainTweenAltOp[T core.Numeric](op any, gradOut, input *core.Tensor[T], times int, lr float64) error {
	times = altTimesOf(times)
	for i := 0; i < times; i++ {
		if err := trainTweenSplitOp(op, gradOut, input, lr); err != nil {
			return fmt.Errorf("tween-alt split %d: %w", i, err)
		}
		if err := applyProjectedTween(op, gradOut, input, lr*0.5); err != nil {
			return fmt.Errorf("tween-alt tween %d: %w", i, err)
		}
	}
	return nil
}
