package embedding

import (
	"fmt"

	"github.com/openfluke/welvet/core"
)

// ApplyGradSGD applies dW (vocab×emb) with SGD on the weight store.
func ApplyGradSGD[T core.Numeric](l *Layer, dW *core.Tensor[T], lr float64) error {
	if l == nil || l.Weights == nil || dW == nil {
		return fmt.Errorf("embedding: ApplyGradSGD nil")
	}
	n := l.GradWSize()
	if dW.Len() < n {
		return fmt.Errorf("embedding: dW len %d < %d", dW.Len(), n)
	}
	buf := make([]float64, n)
	for i := 0; i < n; i++ {
		buf[i] = core.AsFloat64(dW.Data[i])
	}
	return l.Weights.ApplySGD(buf, lr)
}
