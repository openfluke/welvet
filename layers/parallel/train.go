package parallel

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
)

// ApplyGradSGD splits concat dW across branches (+ gate).
func ApplyGradSGD[T core.Numeric](l *Layer, dW *core.Tensor[T], lr float64) error {
	if l == nil {
		return fmt.Errorf("parallel: ApplyGradSGD nil")
	}
	if dW == nil {
		return nil
	}
	off := 0
	for i, ch := range l.Branches {
		n := ch.Weights.Rows * ch.Weights.Cols
		if off+n > dW.Len() {
			return fmt.Errorf("parallel: dW short at branch %d", i)
		}
		slice := core.NewTensor[T](ch.Weights.Rows, ch.Weights.Cols)
		copy(slice.Data, dW.Data[off:off+n])
		if err := dense.ApplyGradSGD(ch, slice, lr); err != nil {
			return err
		}
		off += n
	}
	if l.Gate != nil {
		n := l.Gate.Weights.Rows * l.Gate.Weights.Cols
		if off+n > dW.Len() {
			return fmt.Errorf("parallel: dW short at gate")
		}
		slice := core.NewTensor[T](l.Gate.Weights.Rows, l.Gate.Weights.Cols)
		copy(slice.Data, dW.Data[off:off+n])
		if err := dense.ApplyGradSGD(l.Gate, slice, lr); err != nil {
			return err
		}
	}
	return nil
}
