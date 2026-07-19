package residual

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
)

// ApplyGradSGD applies concatenated F child dWs.
func ApplyGradSGD[T core.Numeric](l *Layer, dW *core.Tensor[T], lr float64) error {
	if l == nil || dW == nil {
		return fmt.Errorf("residual: ApplyGradSGD nil")
	}
	need := l.GradWSize()
	if dW.Len() < need {
		return fmt.Errorf("residual: dW len %d < %d", dW.Len(), need)
	}
	off := 0
	for i, ch := range l.Children {
		n := ch.Weights.Rows * ch.Weights.Cols
		slice := core.NewTensor[T](ch.Weights.Rows, ch.Weights.Cols)
		copy(slice.Data, dW.Data[off:off+n])
		if err := dense.ApplyGradSGD(ch, slice, lr); err != nil {
			return fmt.Errorf("residual child %d SGD: %w", i, err)
		}
		off += n
	}
	return nil
}
