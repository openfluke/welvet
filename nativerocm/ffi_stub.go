//go:build !((linux && amd64 && cgo) || (windows && amd64 && cgo))

package nativerocm

import "fmt"

func openDriver(path string, payload []byte) (driver, error) {
	return nil, fmt.Errorf("nativerocm: Linux/Windows amd64 and cgo are required")
}
