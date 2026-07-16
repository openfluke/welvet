package webgpu

import (
	"fmt"
	"strings"
	"unsafe"

	"github.com/openfluke/webgpu/wgpu"
)

// binGPU holds resident BinaryPacked weight buffers (upload once, GEMV many times).
type binGPU struct {
	scales *wgpu.Buffer
	words  *wgpu.Buffer
	rows   int
	cols   int
	g128   bool
	bytes  uint64
}

func (s *session) ensureBinCache() {
	if s.binCache == nil {
		s.binCache = make(map[uintptr]*binGPU)
	}
}

// ClearBinaryWeightCache releases all sticky BinaryPacked GPU weight buffers.
func ClearBinaryWeightCache() {
	mu.Lock()
	defer mu.Unlock()
	if sess == nil {
		return
	}
	sess.clearBinCacheLocked()
}

func (s *session) clearBinCacheLocked() {
	if s == nil {
		return
	}
	for _, e := range s.binCache {
		if e == nil {
			continue
		}
		if e.scales != nil {
			e.scales.Destroy()
		}
		if e.words != nil {
			e.words.Destroy()
		}
	}
	s.binCache = nil
	s.binCacheBytes = 0
	s.binCacheFull = false
}

// BinaryWeightCacheBytes reports resident BinaryPacked weight VRAM (approx).
func BinaryWeightCacheBytes() uint64 {
	mu.Lock()
	defer mu.Unlock()
	if sess == nil {
		return 0
	}
	return sess.binCacheBytes
}

// WarmBinaryWeight uploads scales+words once and keeps them resident.
// key should be stable (typically BlobKey(unsafe.Pointer(blob))).
func WarmBinaryWeight(key uintptr, scales []float32, words []uint32, rows, cols int, g128 bool) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return initErr
	}
	needWords := (rows * cols) / 32
	needScales := needWords
	if g128 {
		needScales = (rows * cols) / 128
	}
	if len(words) < needWords || len(scales) < needScales {
		return fmt.Errorf("webgpu WarmBinaryWeight: scales/words short")
	}
	mu.Lock()
	defer mu.Unlock()
	_, err := sess.uploadBinaryLocked(key, scales[:needScales], words[:needWords], rows, cols, g128)
	return err
}

func (s *session) uploadBinaryLocked(key uintptr, scales []float32, words []uint32, rows, cols int, g128 bool) (*binGPU, error) {
	s.ensureBinCache()
	if e, ok := s.binCache[key]; ok && e != nil {
		return e, nil
	}
	if s.binCacheFull {
		return nil, errBinaryVRAMFull
	}
	nbytes := uint64(len(scales)*4 + len(words)*4)
	const softCap = 2600 << 20
	if s.binCacheBytes+nbytes > softCap {
		s.binCacheFull = true
		return nil, errBinaryVRAMFull
	}
	dev := s.device
	scBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    "welvet-bin-sc-res",
		Contents: wgpu.ToBytes(scales),
		Usage:    wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("binary scales upload: %w", err)
	}
	wBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    "welvet-bin-w-res",
		Contents: wgpu.ToBytes(words),
		Usage:    wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		scBuf.Destroy()
		return nil, fmt.Errorf("binary words upload: %w", err)
	}
	e := &binGPU{scales: scBuf, words: wBuf, rows: rows, cols: cols, g128: g128, bytes: nbytes}
	s.binCache[key] = e
	s.binCacheBytes += nbytes
	return e, nil
}

var errBinaryVRAMFull = fmt.Errorf("webgpu: binary weight VRAM budget full")

// IsBinaryVRAMFull reports whether further Binary weight uploads should use host.
func IsBinaryVRAMFull(err error) bool {
	if err == nil {
		return false
	}
	if err == errBinaryVRAMFull {
		return true
	}
	s := strings.ToLower(err.Error())
	return strings.Contains(s, "vram") || strings.Contains(s, "not enough memory") || strings.Contains(s, "budget full")
}

// DenseGEMVBinaryResident runs Binary GEMV using sticky GPU weights keyed by key.
// On cache miss, uploads scales/words once. Only x/y move per call after that.
// If key is already resident, scales/words may be nil.
func DenseGEMVBinaryResident(key uintptr, scales []float32, words []uint32, x, y []float32, batch, in, out int, g128 bool) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu DenseGEMVBinaryResident: %w", initErr)
	}
	if key == 0 || in <= 0 || in%32 != 0 || len(x) < batch*in || len(y) < batch*out {
		return fmt.Errorf("webgpu DenseGEMVBinaryResident: shape")
	}
	mu.Lock()
	sess.ensureBinCache()
	if e, ok := sess.binCache[key]; ok && e != nil {
		mu.Unlock()
		return sess.gemvBinaryResident(e, x, y, batch, in, out)
	}
	mu.Unlock()

	needWords := (out * in) / 32
	needScales := needWords
	if g128 {
		if in%128 != 0 {
			return fmt.Errorf("webgpu DenseGEMVBinaryResident: g128 cols")
		}
		needScales = (out * in) / 128
	}
	if len(words) < needWords || len(scales) < needScales {
		return fmt.Errorf("webgpu DenseGEMVBinaryResident: scales/words short")
	}
	mu.Lock()
	e, err := sess.uploadBinaryLocked(key, scales[:needScales], words[:needWords], out, in, g128)
	mu.Unlock()
	if err != nil {
		return err
	}
	return sess.gemvBinaryResident(e, x, y, batch, in, out)
}

// HasBinaryWeight reports whether key is already resident.
func HasBinaryWeight(key uintptr) bool {
	ensure()
	mu.Lock()
	defer mu.Unlock()
	if sess == nil || sess.binCache == nil {
		return false
	}
	_, ok := sess.binCache[key]
	return ok
}

// BlobKey returns a stable cache key for a packed weight blob pointer.
func BlobKey(ptr unsafe.Pointer) uintptr { return uintptr(ptr) }
