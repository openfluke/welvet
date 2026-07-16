package rmsnorm

import (
	"fmt"

	"github.com/openfluke/welvet/core"
)

// ApplyGradSGD applies dγ with SGD.
func ApplyGradSGD[T core.Numeric](l *Layer, dW *core.Tensor[T], lr float64) error {
	if l == nil || l.Gamma == nil || dW == nil {
		return fmt.Errorf("rmsnorm: ApplyGradSGD nil")
	}
	n := l.GradWSize()
	if dW.Len() < n {
		return fmt.Errorf("rmsnorm: dW len %d < %d", dW.Len(), n)
	}
	buf := make([]float64, n)
	for i := 0; i < n; i++ {
		buf[i] = core.AsFloat64(dW.Data[i])
	}
	return l.Gamma.ApplySGD(buf, lr)
}
