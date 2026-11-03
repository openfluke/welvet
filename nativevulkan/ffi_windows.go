//go:build windows && amd64 && cgo

package nativevulkan

import "fmt"

func openDriver(path string, payload []byte) (driver, error) {
	return nil, fmt.Errorf("nativevulkan: Windows Vulkan artifact is not shipped yet")
}
