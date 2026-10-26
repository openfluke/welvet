//go:build !darwin || !arm64 || !cgo

package nativemetal

import "fmt"

func openDriver(path string, payload []byte) (driver, error) {
	return nil, fmt.Errorf("nativemetal: Apple silicon macOS and cgo are required")
}
