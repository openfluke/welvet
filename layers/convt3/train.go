package convt3

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
)

func ApplyGradSGD[T core.Numeric](l *Layer, dW *core.Tensor[T], lr float64) error {
	if l == nil || l.Proj == nil {
		return fmt.Errorf("convt3: ApplyGradSGD nil")
	}
	return dense.ApplyGradSGD(l.Proj, dW, lr)
}
