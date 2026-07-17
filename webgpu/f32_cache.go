package webgpu

import (
	"fmt"
	"strings"
	"unsafe"

	"github.com/openfluke/webgpu/wgpu"
	"github.com/openfluke/welvet/tiling"
)

// f32GPU holds a resident dense FP32 weight matrix (upload once, GEMV many times).
type f32GPU struct {
	w     *wgpu.Buffer
	rows  int
	cols  int
	bytes uint64
}

func (s *session) ensureF32Cache() {
	if s.f32Cache == nil {
		s.f32Cache = make(map[uintptr]*f32GPU)
	}
}

// ClearF32WeightCache releases sticky FP32 weight buffers.
func ClearF32WeightCache() {
	mu.Lock()
	defer mu.Unlock()
	if sess == nil {
		return
	}
	sess.clearF32CacheLocked()
}

func (s *session) clearF32CacheLocked() {
	if s == nil {
		return
	}
	for _, e := range s.f32Cache {
		if e == nil {
			continue
		}
		if e.w != nil {
			e.w.Destroy()
		}
	}
	s.f32Cache = nil
	s.f32CacheBytes = 0
	s.f32CacheFull = false
}

// F32WeightCacheBytes reports resident FP32 weight VRAM.
func F32WeightCacheBytes() uint64 {
	mu.Lock()
	defer mu.Unlock()
	if sess == nil {
		return 0
	}
	return sess.f32CacheBytes
}

// HasF32Weight reports whether key is resident.
func HasF32Weight(key uintptr) bool {
	mu.Lock()
	defer mu.Unlock()
	if sess == nil {
		return false
	}
	e, ok := sess.f32Cache[key]
	return ok && e != nil
}

var errF32VRAMFull = fmt.Errorf("webgpu: F32 weight VRAM budget full")

// IsF32VRAMFull reports a soft-cap warm failure.
func IsF32VRAMFull(err error) bool {
	if err == nil {
		return false
	}
	if err == errF32VRAMFull {
		return true
	}
	s := strings.ToLower(err.Error())
	return strings.Contains(s, "vram") || strings.Contains(s, "not enough memory") || strings.Contains(s, "budget full")
}

// WarmF32Weight uploads a dense [rows*cols] FP32 matrix once.
func WarmF32Weight(key uintptr, w []float32, rows, cols int) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return initErr
	}
	if rows <= 0 || cols <= 0 || len(w) < rows*cols {
		return fmt.Errorf("webgpu WarmF32Weight: shape")
	}
	mu.Lock()
	defer mu.Unlock()
	_, err := sess.uploadF32Locked(key, w[:rows*cols], rows, cols)
	return err
}

func (s *session) uploadF32Locked(key uintptr, w []float32, rows, cols int) (*f32GPU, error) {
	s.ensureF32Cache()
	if e, ok := s.f32Cache[key]; ok && e != nil {
		return e, nil
	}
	if s.f32CacheFull {
		return nil, errF32VRAMFull
	}
	nbytes := uint64(len(w) * 4)
	const softCap = 2800 << 20
	if s.f32CacheBytes+nbytes > softCap {
		s.f32CacheFull = true
		return nil, errF32VRAMFull
	}
	buf, err := s.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    "welvet-f32-w-res",
		Contents: wgpu.ToBytes(w),
		Usage:    wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		msg := strings.ToLower(err.Error())
		if strings.Contains(msg, "memory") || strings.Contains(msg, "oom") || strings.Contains(msg, "out of") {
			s.f32CacheFull = true
			return nil, errF32VRAMFull
		}
		return nil, err
	}
	e := &f32GPU{w: buf, rows: rows, cols: cols, bytes: nbytes}
	s.f32Cache[key] = e
	s.f32CacheBytes += nbytes
	return e, nil
}

// DenseGEMVF32Resident runs FP32 GEMV with sticky weights.
// On cache miss, uploads w once when non-nil.
func DenseGEMVF32Resident(key uintptr, w []float32, x, y []float32, batch, in, out int) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu DenseGEMVF32Resident: %w", initErr)
	}
	if key == 0 || batch <= 0 || in <= 0 || out <= 0 || len(x) < batch*in || len(y) < batch*out {
		return fmt.Errorf("webgpu DenseGEMVF32Resident: shape")
	}
	mu.Lock()
	sess.ensureF32Cache()
	if e, ok := sess.f32Cache[key]; ok && e != nil {
		mu.Unlock()
		return sess.gemvFP32Resident(e, x, y, batch, in, out)
	}
	mu.Unlock()

	if len(w) < out*in {
		return fmt.Errorf("webgpu DenseGEMVF32Resident: missing weights for warm")
	}
	mu.Lock()
	e, err := sess.uploadF32Locked(key, w[:out*in], out, in)
	mu.Unlock()
	if err != nil {
		return err
	}
	return sess.gemvFP32Resident(e, x, y, batch, in, out)
}

func (s *session) gemvFP32Resident(e *f32GPU, x, y []float32, batch, in, out int) error {
	const wg = 64
	dev, q := s.device, s.queue
	if e == nil || e.w == nil {
		return fmt.Errorf("webgpu gemvFP32Resident: nil weight buffer")
	}
	if e.rows != out || e.cols != in {
		return fmt.Errorf("webgpu gemvFP32Resident: cached %dx%d want %dx%d", e.rows, e.cols, out, in)
	}

	xBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-f32-x", Contents: wgpu.ToBytes(x[:batch*in]),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer xBuf.Destroy()

	yBytes := uint64(batch * out * 4)
	if yBytes < 64 {
		yBytes = 64
	}
	yBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "welvet-f32-y", Size: yBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer yBuf.Destroy()

	p := gpuParams{Batch: uint32(batch), InputSize: uint32(in), OutputSize: uint32(out)}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-f32-p", Contents: wgpu.ToBytes([]gpuParams{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()

	if s.pipeFP32 == nil {
		return fmt.Errorf("webgpu: FP32 pipeline missing")
	}
	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.pipeFP32.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: xBuf, Offset: 0, Size: xBuf.GetSize()},
			{Binding: 2, Buffer: e.w, Offset: 0, Size: e.w.GetSize()},
			{Binding: 3, Buffer: yBuf, Offset: 0, Size: yBuf.GetSize()},
		},
	})
	if err != nil {
		return err
	}
	defer bg.Release()

	enc, err := dev.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(s.pipeFP32)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(tiling.GPUWorkgroupsX(out, wg), uint32(batch), 1)
	pass.End()
	cmd, err := enc.Finish(nil)
	if err != nil {
		return err
	}
	q.Submit(cmd)

	outY, err := readbackF32(dev, q, yBuf, batch*out)
	if err != nil {
		return err
	}
	copy(y, outY)
	_ = unsafe.Sizeof(e) // keep e reachable across submit
	return nil
}
