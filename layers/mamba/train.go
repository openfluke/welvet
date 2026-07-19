package mamba

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
)

// ApplyGradSGD applies concat dW: InProj | OutProj | ALog | DSkip.
func ApplyGradSGD[T core.Numeric](l *Layer, dW *core.Tensor[T], lr float64) error {
	if l == nil {
		return fmt.Errorf("mamba: ApplyGradSGD nil")
	}
	if dW == nil {
		return nil
	}
	off := 0
	nIn := l.InProj.Weights.Rows * l.InProj.Weights.Cols
	if off+nIn > dW.Len() {
		return fmt.Errorf("mamba: dW short in")
	}
	sIn := core.NewTensor[T](l.InProj.Weights.Rows, l.InProj.Weights.Cols)
	copy(sIn.Data, dW.Data[off:off+nIn])
	if err := dense.ApplyGradSGD(l.InProj, sIn, lr); err != nil {
		return err
	}
	off += nIn
	nOut := l.OutProj.Weights.Rows * l.OutProj.Weights.Cols
	if off+nOut > dW.Len() {
		return fmt.Errorf("mamba: dW short out")
	}
	sOut := core.NewTensor[T](l.OutProj.Weights.Rows, l.OutProj.Weights.Cols)
	copy(sOut.Data, dW.Data[off:off+nOut])
	if err := dense.ApplyGradSGD(l.OutProj, sOut, lr); err != nil {
		return err
	}
	off += nOut
	inner := l.Cfg.InnerDim()
	if off+2*inner > dW.Len() {
		return fmt.Errorf("mamba: dW short A/D")
	}
	for i := 0; i < inner; i++ {
		l.ALog[i] -= float32(lr * core.AsFloat64(dW.Data[off+i]))
		l.DSkip[i] -= float32(lr * core.AsFloat64(dW.Data[off+inner+i]))
	}
	return nil
}
