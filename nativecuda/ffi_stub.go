//go:build !linux || !amd64 || !cgo

package nativecuda

import "fmt"

func openDriver(path string, payload []byte) (driver, error) {
	return nil, fmt.Errorf("nativecuda: Linux amd64 and cgo are required")
}
