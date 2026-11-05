package nativemlx

import (
	"fmt"
	"sync"
)

// Shared GPU session for layer ForwardMLX calls (lazy open).
var (
	sessMu   sync.Mutex
	shared   *Session
	sessErr  error
	sessOnce sync.Once
)

// DefaultSession returns a process-wide MLX GPU session (opened once).
func DefaultSession() (*Session, error) {
	sessOnce.Do(func() {
		if !Available() {
			sessErr = errUnavailable()
			return
		}
		shared, sessErr = OpenGPU()
	})
	if sessErr != nil {
		return nil, sessErr
	}
	if shared == nil || !shared.Ready() {
		return nil, fmt.Errorf("nativemlx: GPU session not ready")
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
	return fmt.Errorf("nativemlx: libwelvet_mlx not available (need cgo + darwin/arm64 + shipped dylib + Metal GPU)")
}

// RequireLayer rejects mounts the MLX caps table does not list.
func RequireLayer(layer, dtype string) error {
	if !Available() {
		return errUnavailable()
	}
	if !LayerSupported(layer, dtype) {
		return fmt.Errorf("nativemlx: reject %s/%s — not supported on MLX GPU (see SupportedLayers)", layer, dtype)
	}
	return nil
}

// DTypeName maps common Welvet dtype names to welvet.mlx caps strings.
func DTypeName(dt string) string {
	switch dt {
	case "Float32", "float32", "f32", "FLOAT32":
		return "float32"
	case "Float16", "float16", "f16", "FLOAT16":
		return "float16"
	case "BFloat16", "bfloat16", "bf16", "BFLOAT16":
		return "bfloat16"
	case "Float64", "float64", "f64", "FLOAT64":
		return "float64"
	case "Int8", "int8", "i8":
		return "int8"
	case "Int16", "int16", "i16":
		return "int16"
	case "Int32", "int32", "i32":
		return "int32"
	case "Int64", "int64", "i64":
		return "int64"
	case "UInt8", "Uint8", "uint8", "u8":
		return "uint8"
	case "UInt16", "Uint16", "uint16", "u16":
		return "uint16"
	case "UInt32", "Uint32", "uint32", "u32":
		return "uint32"
	case "UInt64", "Uint64", "uint64", "u64":
		return "uint64"
	case "Complex64", "complex64", "c64":
		return "complex64"
	default:
		return "float32"
	}
}
