package layernorm

import (
	"fmt"

	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/quant"
)

// SyncToNative pushes γ|β into the native mirror (user-owned).
func (l *Layer) SyncToNative() error {
	if l == nil || l.Gamma == nil || l.Beta == nil {
		return fmt.Errorf("layernorm: SyncToNative nil")
	}
	if l.Gamma.Format != quant.FormatNone || l.Beta.Format != quant.FormatNone {
		return fmt.Errorf("layernorm: SyncToNative needs FormatNone")
	}
	if !nativecpp.NativeCPUAvailable() {
		return fmt.Errorf("layernorm: SyncToNative needs libwelvet")
	}
	g, err := l.Gamma.FlattenF32()
	if err != nil {
		return err
	}
	b, err := l.Beta.FlattenF32()
	if err != nil {
		return err
	}
	concat := append(append([]float32(nil), g...), b...)
	if l.nativeMir == nil {
		l.nativeMir = &nativecpp.Mirror{}
	}
	l.nativeMir.Capture(concat, l.Gamma, l.Beta)
	return nil
}
