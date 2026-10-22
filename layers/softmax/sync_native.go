package softmax

import (
	"fmt"

	"github.com/openfluke/welvet/nativecpp"
)

// SyncToNative is a no-op weight push for weightless Softmax (caps / availability only).
func (l *Layer) SyncToNative() error {
	if l == nil {
		return fmt.Errorf("softmax: SyncToNative nil")
	}
	if !nativecpp.NativeCPUAvailable() {
		return fmt.Errorf("softmax: SyncToNative needs libwelvet")
	}
	return nil
}
