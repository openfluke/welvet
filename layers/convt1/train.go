package convt1

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
)

// ApplyGradSGD applies dW via the Dense projection store.
func ApplyGradSGD[T core.Numeric](l *Layer, dW *core.Tensor[T], lr float64) error {
	if l == nil || l.Proj == nil {
		return fmt.Errorf("convt1: ApplyGradSGD nil")
	}
	return dense.ApplyGradSGD(l.Proj, dW, lr)
}
