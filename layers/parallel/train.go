package parallel

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
)

// ApplyGradSGD splits concat dW across branches (+ gate) using each Op's GradWSize.
func ApplyGradSGD[T core.Numeric](l *Layer, dW *core.Tensor[T], lr float64) error {
	if l == nil {
		return fmt.Errorf("parallel: ApplyGradSGD nil")
	}
	if dW == nil {
		return nil
	}
	off := 0
	for i, ch := range l.Branches {
		n := branchGradWSize(ch)
		if n == 0 {
			continue
		}
		if off+n > dW.Len() {
			return fmt.Errorf("parallel: dW short at branch %d (need %d, have %d)", i, off+n, dW.Len())
		}
		slice := core.NewTensor[T](n)
		copy(slice.Data, dW.Data[off:off+n])
		if err := branchApplyGradSGD(ch, slice, lr); err != nil {
			return fmt.Errorf("parallel branch %d SGD: %w", i, err)
		}
		off += n
	}
	if l.Gate != nil {
		n := l.Gate.GradWSize()
		if n > 0 {
			if off+n > dW.Len() {
				return fmt.Errorf("parallel: dW short at gate")
			}
			slice := core.NewTensor[T](n)
			copy(slice.Data, dW.Data[off:off+n])
			if err := dense.ApplyGradSGD(l.Gate, slice, lr); err != nil {
				return err
			}
		}
	}
	return nil
}
