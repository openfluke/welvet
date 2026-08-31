package parallel

import (
	"fmt"
	"math"

	"github.com/openfluke/welvet/core"
)

// outputGap maps stack logits + one-hot target onto a loss and a gy the
// credit modes already know how to walk (BP / Tween / Split / proxy / Sparse).
type outputGap[T core.Numeric] func(pred, target *core.Tensor[T]) (float64, *core.Tensor[T], error)

// TrainStackMSE runs ForwardStack → MSE → TrainStack under mode.
// Split-family modes use one collect tape (OpenSplitTape). Alt recomputes
// the MSE gap between Split and Tween phases.
func TrainStackMSE[T core.Numeric](s *Stack, input, target *core.Tensor[T], mode TrainMode, lr float64) (loss float64, err error) {
	return trainStackGap(s, input, target, mode, lr, mseGrad[T], trainTweenAltStackMSE[T])
}

// TrainStackCE is next-char / class training: softmax(logits) − one-hot, then
// the same TrainStack credit walk as MSE. Mean over batch, not over B×V.
func TrainStackCE[T core.Numeric](s *Stack, input, target *core.Tensor[T], mode TrainMode, lr float64) (loss float64, err error) {
	return trainStackGap(s, input, target, mode, lr, ceGrad[T], trainTweenAltStackCE[T])
}

func trainStackGap[T core.Numeric](s *Stack, input, target *core.Tensor[T], mode TrainMode, lr float64, gap outputGap[T], alt func(*Stack, *core.Tensor[T], *core.Tensor[T], float64) (float64, error)) (loss float64, err error) {
	if s == nil || input == nil || target == nil {
		return 0, fmt.Errorf("parallel: TrainStack nil")
	}
	defer func() {
		if err != nil || s == nil {
			return
		}
		_ = s.MaybeSync(SyncAfterStep)
		_ = s.MaybeSync(SyncAfterSample)
	}()
	if mode.IsLineStep() {
		return trainStackLine(s, input, target, mode, lr, gap)
	}
	if mode.Resolve(ModeNormalBP).Family() == familyTweenAlt {
		return alt(s, input, target, lr)
	}
	if mode.Resolve(ModeNormalBP).Family() == familyTweenSplit {
		tape, errOpen := OpenSplitTape(s, input)
		if errOpen != nil {
			return 0, errOpen
		}
		return tape.TrainGap(target, mode, lr, gap)
	}
	pre, post, errFwd := ForwardStack(s, input)
	if errFwd != nil {
		return 0, errFwd
	}
	loss, gy, errGap := gap(post, target)
	if errGap != nil {
		return 0, errGap
	}
	if errTrain := TrainStack(s, gy, input, pre, mode, lr); errTrain != nil {
		return loss, errTrain
	}
	return loss, nil
}

// TrainMSE runs Forward → MSE → Train under mode (Parallel root).
func TrainMSE[T core.Numeric](l *Layer, input, target *core.Tensor[T], mode TrainMode, lr float64) (loss float64, err error) {
	if l == nil || input == nil || target == nil {
		return 0, fmt.Errorf("parallel: TrainMSE nil")
	}
	if mode.Resolve(ModeNormalBP).Family() == familyTweenAlt {
		return trainTweenAltLayerMSE(l, input, target, lr)
	}
	if mode.Resolve(ModeNormalBP).Family() == familyTweenSplit {
		tape, err := OpenSplitTape(l, input)
		if err != nil {
			return 0, err
		}
		return tape.Train(target, mode, lr)
	}
	pre, post, err := Forward(l, input)
	if err != nil {
		return 0, err
	}
	loss, gy, err := mseGrad(post, target)
	if err != nil {
		return 0, err
	}
	l.NoteLoss(loss)
	if err := Train(l, gy, input, pre, mode, lr); err != nil {
		return loss, err
	}
	_ = l.MaybeSync(SyncAfterStep)
	_ = l.MaybeSync(SyncAfterSample)
	l.AdvanceRotate()
	l.RefreshMetrics()
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

// ceGrad is softmax-CE vs one-hot. gy = (p − t) / B on [B, C] logits.
func ceGrad[T core.Numeric](pred, target *core.Tensor[T]) (loss float64, gy *core.Tensor[T], err error) {
	if pred == nil || target == nil || pred.Len() != target.Len() {
		return 0, nil, fmt.Errorf("parallel: ce shape mismatch")
	}
	if len(pred.Shape) < 1 {
		return 0, nil, fmt.Errorf("parallel: ce empty shape")
	}
	classes := pred.Shape[len(pred.Shape)-1]
	if classes < 2 || pred.Len()%classes != 0 {
		return 0, nil, fmt.Errorf("parallel: ce classes %d len %d", classes, pred.Len())
	}
	batch := pred.Len() / classes
	gy = core.NewTensor[T](pred.Shape...)
	invB := 1.0 / float64(batch)
	var nll float64
	probs := make([]float64, classes)
	for b := 0; b < batch; b++ {
		off := b * classes
		max := core.AsFloat64(pred.Data[off])
		for c := 1; c < classes; c++ {
			if v := core.AsFloat64(pred.Data[off+c]); v > max {
				max = v
			}
		}
		sum := 0.0
		for c := 0; c < classes; c++ {
			probs[c] = math.Exp(core.AsFloat64(pred.Data[off+c]) - max)
			sum += probs[c]
		}
		inv := 1.0 / sum
		for c := 0; c < classes; c++ {
			p := probs[c] * inv
			t := core.AsFloat64(target.Data[off+c])
			if t > 0 {
				pp := p
				if pp < 1e-12 {
					pp = 1e-12
				}
				nll -= t * math.Log(pp)
			}
			gy.Data[off+c] = core.FromFloat64[T]((p - t) * invB)
		}
	}
	return nll / float64(batch), gy, nil
}
