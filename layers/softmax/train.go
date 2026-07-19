package softmax

import (
	"fmt"

	"github.com/openfluke/welvet/core"
)

// ApplyGradSGD is a no-op (weightless). Accepts nil dW.
func ApplyGradSGD[T core.Numeric](l *Layer, dW *core.Tensor[T], lr float64) error {
	if l == nil {
		return fmt.Errorf("softmax: ApplyGradSGD nil layer")
	}
	_ = dW
	_ = lr
	return nil
}
