package kmeans

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
)

func ApplyGradSGD[T core.Numeric](l *Layer, dW *core.Tensor[T], lr float64) error {
	if l == nil || l.Centers == nil {
		return fmt.Errorf("kmeans: ApplyGradSGD nil")
	}
	return dense.ApplyGradSGD(l.Centers, dW, lr)
}
