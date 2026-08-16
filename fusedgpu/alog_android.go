//go:build birdkit_native_vk && android && cgo

package fusedgpu

// ALog writes to logcat tag birdkit-vk (visible in flutter run).
func ALog(msg string) { vkALog(msg) }
