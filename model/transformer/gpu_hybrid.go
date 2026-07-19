package transformer

import (
	"fmt"
	"sort"
	"unsafe"

	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/webgpu"
)

// minBinaryGPUBytes: below this, host BinaryG128 is faster than a WebGPU dispatch.
const minBinaryGPUBytes = 512 << 10 // 512 KiB packed

// SyncHybridGPU uploads BinaryPacked projections once (largest first) and keeps
// them resident. Tiny tensors stay on host — GPU dispatch overhead dominates there.
func (m *Model) SyncHybridGPU() error {
	if m == nil {
		return fmt.Errorf("transformer: nil model")
	}
	if !webgpu.Available() {
		if err := webgpu.InitError(); err != nil {
			return err
		}
		return fmt.Errorf("webgpu: no adapter")
	}
	webgpu.ClearBinaryWeightCache()

	var blobs []*quant.Blob
	add := func(b *quant.Blob) {
		if b == nil || b.Format != quant.FormatBinaryPacked {
			return
		}
		if len(b.Raw) < minBinaryGPUBytes {
			return // keep on CPU
		}
		blobs = append(blobs, b)
	}
	add(m.lmHeadPacked)
	if m.lmHead != nil {
		add(m.lmHead.Packed)
	}
	for i := range m.Blocks {
		b := &m.Blocks[i]
		for _, d := range []*dense.Layer{b.Q, b.K, b.V, b.O} {
			if d != nil && d.Weights != nil {
				add(d.Weights.Packed)
			}
		}
		if b.FFN != nil {
			for _, d := range []*dense.Layer{b.FFN.Gate, b.FFN.Up, b.FFN.Down} {
				if d != nil && d.Weights != nil {
					add(d.Weights.Packed)
				}
			}
		}
		if b.GDN != nil {
			add(b.GDN.InQKV)
			add(b.GDN.InZ)
			add(b.GDN.InB)
			add(b.GDN.InA)
			add(b.GDN.Out)
		}
	}
	// Largest first so FFN/attn fill VRAM before small leftovers.
	sort.Slice(blobs, func(i, j int) bool {
		return len(blobs[i].Raw) > len(blobs[j].Raw)
	})

	warmed := 0
	skipped := 0
	for _, b := range blobs {
		if err := warmBinaryBlob(b); err != nil {
			if webgpu.IsBinaryVRAMFull(err) {
				skipped += len(blobs) - warmed - skipped
				break
			}
			// Single tensor too big / transient OOM — try next smaller one.
			skipped++
			continue
		}
		warmed++
	}
	mb := float64(webgpu.BinaryWeightCacheBytes()) / (1024 * 1024)
	fmt.Printf("  hybrid GPU: resident BinaryG128 %d tensors (%.0f MiB) — largest-first\n", warmed, mb)
	if warmed == 0 {
		fmt.Println("  note: could not pin weights in VRAM — GEMVs use host BinaryG128 (try free GPU memory / 8GB+ card)")
		return nil
	}
	if skipped > 0 || warmed < len(blobs) {
		fmt.Printf("  note: %d tensors on host (VRAM/size); ~8GB+ GPU for full residency\n", len(blobs)-warmed)
	}
	return nil
}

func warmBinaryBlob(b *quant.Blob) error {
	scales, words, g128, ok := dense.BinaryBlobStaging(b)
	if !ok {
		return fmt.Errorf("binary staging failed")
	}
	return webgpu.WarmBinaryWeight(webgpu.BlobKey(unsafe.Pointer(b)), scales, words, b.Rows, b.Cols, g128)
}

// CloseHybridGPU drops sticky BinaryPacked GPU weights.
func (m *Model) CloseHybridGPU() {
	webgpu.ClearBinaryWeightCache()
}
