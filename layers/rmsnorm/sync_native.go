package rmsnorm

import (
	"fmt"

	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/quant"
)

// SyncToNative pushes γ into the native mirror (user-owned).
func (l *Layer) SyncToNative() error {
	if l == nil || l.Gamma == nil {
		return fmt.Errorf("rmsnorm: SyncToNative nil")
	}
	if l.Gamma.Format != quant.FormatNone {
		return fmt.Errorf("rmsnorm: SyncToNative needs FormatNone")
	}
	if !nativecpp.NativeCPUAvailable() {
		return fmt.Errorf("rmsnorm: SyncToNative needs libwelvet")
	}
	g, err := l.Gamma.FlattenF32()
	if err != nil {
		return err
	}
	if l.nativeMir == nil {
		l.nativeMir = &nativecpp.Mirror{}
	}
	l.nativeMir.Capture(g, l.Gamma)
	return nil
}
