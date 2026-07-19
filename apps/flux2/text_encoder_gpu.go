package flux2

import (
	"fmt"
	"sort"
	"unsafe"

	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/webgpu"
)

const minAffineGPUBytes = 64 << 10 // 64 KiB

// SyncGPU uploads Affine4 linears (layers used for Klein stack) to VRAM.
// Call before EncodeHiddenStates; CloseGPU afterward so Flux2 BinaryG128 can use VRAM.
func (m *Qwen3TextEncoder) SyncGPU(maxLayer int) error {
	if m == nil {
		return fmt.Errorf("Qwen3TextEncoder.SyncGPU: nil")
	}
	if !webgpu.Available() {
		if err := webgpu.InitError(); err != nil {
			return err
		}
		return fmt.Errorf("webgpu: no adapter")
	}
	if maxLayer < 0 {
		maxLayer = 26 // Klein needs hidden_states[27] → layers 0..26
	}
	if maxLayer >= len(m.Layers) {
		maxLayer = len(m.Layers) - 1
	}
	webgpu.ClearAffineWeightCache()

	type item struct {
		b    *quant.Blob
		bytes int
	}
	seen := map[uintptr]struct{}{}
	var blobs []item
	add := func(l *Linear) {
		if l == nil || l.Blob == nil || !quant.IsAffinePacked(l.Blob) {
			return
		}
		if len(l.Blob.Raw) < minAffineGPUBytes {
			return
		}
		k := uintptr(unsafe.Pointer(l.Blob))
		if _, ok := seen[k]; ok {
			return
		}
		seen[k] = struct{}{}
		blobs = append(blobs, item{b: l.Blob, bytes: len(l.Blob.Raw)})
	}
	for i := 0; i <= maxLayer; i++ {
		layer := &m.Layers[i]
		add(layer.Q)
		add(layer.K)
		add(layer.V)
		add(layer.O)
		add(layer.Gate)
		add(layer.Up)
		add(layer.Down)
	}
	sort.Slice(blobs, func(i, j int) bool { return blobs[i].bytes > blobs[j].bytes })

	warmed := 0
	for _, it := range blobs {
		scales, biases, words, group, ok := dense.AffineBlobStaging(it.b)
		if !ok {
			continue
		}
		err := webgpu.WarmAffineWeight(webgpu.BlobKey(unsafe.Pointer(it.b)), scales, biases, words, it.b.Rows, it.b.Cols, group)
		if err != nil {
			if webgpu.IsAffineVRAMFull(err) {
				break
			}
			continue
		}
		warmed++
	}
	mb := float64(webgpu.AffineWeightCacheBytes()) / (1024 * 1024)
	fmt.Printf("  text-enc GPU: resident Affine4 %d tensors (%.0f MiB)\n", warmed, mb)
	if warmed == 0 {
		m.UseGPU = false
		m.setLinearsGPU(false)
		return fmt.Errorf("Qwen3 SyncGPU: could not pin any Affine4 in VRAM")
	}
	m.UseGPU = true
	m.setLinearsGPU(true)
	return nil
}

// CloseGPU drops Affine4 resident weights.
func (m *Qwen3TextEncoder) CloseGPU() {
	if m == nil {
		return
	}
	webgpu.ClearAffineWeightCache()
	m.UseGPU = false
	m.setLinearsGPU(false)
}

func (m *Qwen3TextEncoder) setLinearsGPU(on bool) {
	if m == nil {
		return
	}
	for i := range m.Layers {
		layer := &m.Layers[i]
		for _, l := range []*Linear{layer.Q, layer.K, layer.V, layer.O, layer.Gate, layer.Up, layer.Down} {
			if l == nil || l.Blob == nil || !quant.IsAffinePacked(l.Blob) {
				continue
			}
			l.UseGPU = on && webgpu.HasAffineWeight(webgpu.BlobKey(unsafe.Pointer(l.Blob)))
		}
	}
}
