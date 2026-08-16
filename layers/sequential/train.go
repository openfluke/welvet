package sequential

import (
	"fmt"

	"github.com/openfluke/welvet/core"
)

// ApplyGradSGD applies concatenated child dWs (child0 || child1 || …).
func ApplyGradSGD[T core.Numeric](l *Layer, dW *core.Tensor[T], lr float64) error {
	if l == nil || dW == nil {
		return fmt.Errorf("sequential: ApplyGradSGD nil")
	}
	need := l.GradWSize()
	if dW.Len() < need {
		return fmt.Errorf("sequential: dW len %d < %d", dW.Len(), need)
	}
	off := 0
	for i, op := range l.ChildOps() {
		n := childGradWSize(op)
		if n == 0 {
			continue
		}
		slice := core.NewTensor[T](n)
		copy(slice.Data, dW.Data[off:off+n])
		if err := childApplySGD(op, slice, lr); err != nil {
			return fmt.Errorf("sequential child %d SGD: %w", i, err)
		}
		off += n
	}
	return nil
}
