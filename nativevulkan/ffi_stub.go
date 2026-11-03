//go:build !((linux && amd64 && cgo) || (windows && amd64 && cgo))

package nativevulkan

import "fmt"

func openDriver(path string, payload []byte) (driver, error) {
	return nil, fmt.Errorf("nativevulkan: Linux amd64 and cgo are required")
}
