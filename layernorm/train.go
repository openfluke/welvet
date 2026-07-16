package layernorm

import (
	"fmt"

	"github.com/openfluke/welvet/core"
)

// ApplyGradSGD applies [dγ|dβ] with SGD.
func ApplyGradSGD[T core.Numeric](l *Layer, dW *core.Tensor[T], lr float64) error {
	if l == nil || l.Gamma == nil || l.Beta == nil || dW == nil {
		return fmt.Errorf("layernorm: ApplyGradSGD nil")
	}
	n := l.GradWSize()
	if dW.Len() < n {
		return fmt.Errorf("layernorm: dW len %d < %d", dW.Len(), n)
	}
	dim := l.Cfg.Dim
	gBuf := make([]float64, dim)
	bBuf := make([]float64, dim)
	for i := 0; i < dim; i++ {
		gBuf[i] = core.AsFloat64(dW.Data[i])
		bBuf[i] = core.AsFloat64(dW.Data[dim+i])
	}
	if err := l.Gamma.ApplySGD(gBuf, lr); err != nil {
		return err
	}
	return l.Beta.ApplySGD(bBuf, lr)
}
