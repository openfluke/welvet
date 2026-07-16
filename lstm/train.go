package lstm

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/dense"
)

// ApplyGradSGD applies loom-packed [i|f|g|o] gate grads.
func ApplyGradSGD[T core.Numeric](l *Layer, dW *core.Tensor[T], lr float64) error {
	if l == nil || dW == nil {
		return fmt.Errorf("lstm: ApplyGradSGD nil")
	}
	need := l.GradWSize()
	if dW.Len() < need {
		return fmt.Errorf("lstm: dW len %d < %d", dW.Len(), need)
	}
	gateN := l.Cfg.GateSize()
	ihN := l.I.IH.Weights.Rows * l.I.IH.Weights.Cols
	hhN := l.I.HH.Weights.Rows * l.I.HH.Weights.Cols
	gates := l.gates()
	for gi, g := range gates {
		off := gi * gateN
		ihSlice := core.NewTensor[T](g.IH.Weights.Rows, g.IH.Weights.Cols)
		copy(ihSlice.Data, dW.Data[off:off+ihN])
		if err := dense.ApplyGradSGD(g.IH, ihSlice, lr); err != nil {
			return err
		}
		hhSlice := core.NewTensor[T](g.HH.Weights.Rows, g.HH.Weights.Cols)
		copy(hhSlice.Data, dW.Data[off+ihN:off+ihN+hhN])
		if err := dense.ApplyGradSGD(g.HH, hhSlice, lr); err != nil {
			return err
		}
		dB := make([]float64, l.Cfg.HiddenSize)
		for i := 0; i < l.Cfg.HiddenSize; i++ {
			dB[i] = core.AsFloat64(dW.Data[off+ihN+hhN+i])
		}
		if err := g.IH.Weights.ApplyBiasSGD(dB, lr); err != nil {
			return err
		}
	}
	return nil
}
