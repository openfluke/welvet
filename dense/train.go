package dense

import (
	"fmt"

	"github.com/openfluke/welvet/core"
)

// ApplyGradSGD applies dW (and optional per-row bias grad) with SGD.
// Future layers mirror this method name so training can dispatch uniformly.
func ApplyGradSGD[T core.Numeric](l *Layer, dW *core.Tensor[T], lr float64) error {
	if l == nil || l.Weights == nil || dW == nil {
		return fmt.Errorf("dense: ApplyGradSGD nil")
	}
	n := l.Weights.Rows * l.Weights.Cols
	if dW.Len() < n {
		return fmt.Errorf("dense: dW len %d < %d", dW.Len(), n)
	}
	buf := make([]float64, n)
	for i := 0; i < n; i++ {
		buf[i] = core.AsFloat64(dW.Data[i])
	}
	return l.Weights.ApplySGD(buf, lr)
}
