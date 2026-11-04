//go:build !(((linux && amd64) || (android && (amd64 || arm64)) || (windows && amd64)) && cgo)

package nativevulkan

import "fmt"

func openDriver(path string, payload []byte) (driver, error) {
	return nil, fmt.Errorf("nativevulkan: Linux/Android amd64|arm64 (or Windows amd64) with cgo required")
}
