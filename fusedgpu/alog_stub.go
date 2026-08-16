//go:build !birdkit_native_vk || !android || !cgo

package fusedgpu

import "fmt"

// ALog prints to stdout when android logcat bridge is unavailable.
func ALog(msg string) { fmt.Println(msg) }
