package sequential

import (
	"fmt"

	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/nativecpp"
)

// SyncToNative pushes each child into its native mirror (user-owned).
func (l *Layer) SyncToNative() error {
	if l == nil {
		return fmt.Errorf("sequential: SyncToNative nil")
	}
	if !nativecpp.NativeCPUAvailable() {
		return fmt.Errorf("sequential: SyncToNative needs libwelvet")
	}
	for i, op := range l.ChildOps() {
		if err := syncChildToNative(op); err != nil {
			return fmt.Errorf("sequential: SyncToNative child %d: %w", i, err)
		}
	}
	return nil
}

func syncChildToNative(op any) error {
	switch v := op.(type) {
	case *dense.Layer:
		return v.SyncToNative()
	case *rmsnorm.Layer:
		return v.SyncToNative()
	case *layernorm.Layer:
		return v.SyncToNative()
	default:
		return fmt.Errorf("no SyncToNative on %T", op)
	}
}
