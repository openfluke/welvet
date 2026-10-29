package cnn2

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
)

// ApplyGradSGD applies dW via the Dense projection store.
func ApplyGradSGD[T core.Numeric](l *Layer, dW *core.Tensor[T], lr float64) error {
	if l != nil && l.Exec.Backend == core.BackendMetal {
		return core.MetalApplySGD(l, dW, lr)
	}
	if l != nil && l.Exec.Backend == core.BackendCuda {
		return core.CudaApplySGD(l, dW, lr)
	}
	if l == nil || l.Proj == nil {
		return fmt.Errorf("cnn2: ApplyGradSGD nil")
	}
	return dense.ApplyGradSGD(l.Proj, dW, lr)
}
