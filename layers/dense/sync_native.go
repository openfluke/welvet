package dense

import (
	"fmt"

	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/weights"
)

// SyncToNative pushes Go Weights into the native mirror (user-owned).
// Required before fair NativeCPU timing; ForwardNativeCPU then reads NativeMatrixF32.
func (l *Layer) SyncToNative() error {
	if l == nil || l.Weights == nil {
		return fmt.Errorf("dense: SyncToNative nil")
	}
	if l.Weights.Format != quant.FormatNone {
		return fmt.Errorf("dense: SyncToNative needs FormatNone")
	}
	if !nativecpp.NativeCPUAvailable() {
		return fmt.Errorf("dense: SyncToNative needs libwelvet")
	}
	w, err := weights.MatrixF32(l.Weights)
	if err != nil {
		return err
	}
	l.EnsureNativeMirror().Capture(w, l.Weights)
	return nil
}

// NativeSynced reports whether SyncToNative captured a pending buffer.
func (l *Layer) NativeSynced() bool {
	return l != nil && l.nativeMir != nil && l.nativeMir.Dirty()
}
