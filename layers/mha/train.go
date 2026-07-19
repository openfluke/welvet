package mha

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
)

// ApplyGradSGD applies concatenated dW (Q then K then V then O) with SGD.
func ApplyGradSGD[T core.Numeric](l *Layer, dW *core.Tensor[T], lr float64) error {
	if l == nil || dW == nil {
		return fmt.Errorf("mha: ApplyGradSGD nil")
	}
	need := l.GradWSize()
	if dW.Len() < need {
		return fmt.Errorf("mha: dW len %d < %d", dW.Len(), need)
	}
	off := 0
	for _, p := range []*dense.Layer{l.Q, l.K, l.V, l.O} {
		n := p.Weights.Rows * p.Weights.Cols
		slice := core.NewTensor[T](p.Weights.Rows, p.Weights.Cols)
		copy(slice.Data, dW.Data[off:off+n])
		if err := dense.ApplyGradSGD(p, slice, lr); err != nil {
			return err
		}
		off += n
	}
	return nil
}
