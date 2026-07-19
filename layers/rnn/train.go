package rnn

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
)

// ApplyGradSGD applies loom-packed [dIH | dHH | dBias].
func ApplyGradSGD[T core.Numeric](l *Layer, dW *core.Tensor[T], lr float64) error {
	if l == nil || l.IH == nil || l.HH == nil || dW == nil {
		return fmt.Errorf("rnn: ApplyGradSGD nil")
	}
	need := l.GradWSize()
	if dW.Len() < need {
		return fmt.Errorf("rnn: dW len %d < %d", dW.Len(), need)
	}
	ihN := l.IH.Weights.Rows * l.IH.Weights.Cols
	hhN := l.HH.Weights.Rows * l.HH.Weights.Cols
	ihSlice := core.NewTensor[T](l.IH.Weights.Rows, l.IH.Weights.Cols)
	copy(ihSlice.Data, dW.Data[:ihN])
	if err := dense.ApplyGradSGD(l.IH, ihSlice, lr); err != nil {
		return err
	}
	hhSlice := core.NewTensor[T](l.HH.Weights.Rows, l.HH.Weights.Cols)
	copy(hhSlice.Data, dW.Data[ihN:ihN+hhN])
	if err := dense.ApplyGradSGD(l.HH, hhSlice, lr); err != nil {
		return err
	}
	dB := make([]float64, l.Cfg.HiddenSize)
	for i := 0; i < l.Cfg.HiddenSize; i++ {
		dB[i] = core.AsFloat64(dW.Data[ihN+hhN+i])
	}
	return l.IH.Weights.ApplyBiasSGD(dB, lr)
}
