package nativeqnn

import (
	"fmt"
	"sync"
)

// Shared HTP session for layer ForwardQNN calls (lazy open).
var (
	sessMu   sync.Mutex
	shared   *Session
	sessErr  error
	sessOnce sync.Once
)

// DefaultSession returns a process-wide HTP session (opened once).
func DefaultSession() (*Session, error) {
	sessOnce.Do(func() {
		if !Available() {
			sessErr = errUnavailable()
			return
		}
		shared, sessErr = OpenHTP()
	})
	if sessErr != nil {
		return nil, sessErr
	}
	if shared == nil || !shared.Ready() {
		return nil, fmt.Errorf("nativeqnn: HTP session not ready")
	}
	return shared, nil
}

// CloseDefaultSession closes the shared session (tests / shutdown).
func CloseDefaultSession() {
	sessMu.Lock()
	defer sessMu.Unlock()
	if shared != nil {
		shared.Close()
		shared = nil
	}
	sessOnce = sync.Once{}
	sessErr = nil
}

func errUnavailable() error {
	return fmt.Errorf("nativeqnn: libwelvet_qnn not available (need cgo + windows/arm64 + shipped DLL)")
}

// RequireLayer rejects mounts the HTP caps table does not list.
func RequireLayer(layer, dtype string) error {
	if !Available() {
		return errUnavailable()
	}
	if !LayerSupported(layer, dtype) {
		return fmt.Errorf("nativeqnn: reject %s/%s — not supported on HTP (see SupportedLayers)", layer, dtype)
	}
	return nil
}

// DTypeName maps common Welvet dtype names to welvet.qnn caps strings.
func DTypeName(dt string) string {
	switch dt {
	case "Float32", "float32", "f32", "FLOAT32":
		return "float32"
	case "Float16", "float16", "f16", "FLOAT16":
		return "float16"
	case "BFloat16", "bfloat16", "bf16", "BFLOAT16":
		return "bfloat16"
	case "Int8", "int8", "i8":
		return "int8"
	case "UInt8", "uint8", "u8":
		return "uint8"
	default:
		return "float32"
	}
}
