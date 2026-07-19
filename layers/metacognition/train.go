package metacognition

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
)

func ApplyGradSGD[T core.Numeric](l *Layer, dW *core.Tensor[T], lr float64) error {
	if l == nil || l.Observed == nil {
		return fmt.Errorf("metacognition: ApplyGradSGD nil")
	}
	return dense.ApplyGradSGD(l.Observed, dW, lr)
}
