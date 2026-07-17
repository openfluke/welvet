package webgpu

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
)

// WithDevice runs fn with exclusive access to the shared WebGPU device/queue.
// Do not call other webgpu entry points from fn (they also take the session lock).
func WithDevice(fn func(dev *wgpu.Device, queue *wgpu.Queue) error) error {
	ensure()
	mu.Lock()
	defer mu.Unlock()
	if !haveGPU || sess == nil {
		if initErr != nil {
			return initErr
		}
		return fmt.Errorf("webgpu: no device")
	}
	return fn(sess.device, sess.queue)
}

// ReadbackF32 copies count float32s from a device storage buffer (CopySrc) to host.
// Caller must hold the session via WithDevice (or otherwise serialize GPU access).
func ReadbackF32(dev *wgpu.Device, q *wgpu.Queue, buf *wgpu.Buffer, count int) ([]float32, error) {
	return readbackF32(dev, q, buf, count)
}

// MakeComputePipeline compiles WGSL with entry point "main".
func MakeComputePipeline(dev *wgpu.Device, code, label string) (*wgpu.ComputePipeline, error) {
	return makePipeline(dev, code, label)
}
