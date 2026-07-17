package flux2

import (
	"fmt"
	"sort"
	"unsafe"

	"github.com/openfluke/welvet/dense"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/webgpu"
)

// minBinaryGPUBytes: below this, host BinaryG128 beats a WebGPU round-trip.
const minBinaryGPUBytes = 512 << 10 // 512 KiB packed

// SyncGPU uploads BinaryG128 Flux2 linears to VRAM (largest first) and enables
// batched GEMV on those weights. Attention / norms / dense skips stay on host.
func (m *Model) SyncGPU() error {
	if m == nil {
		return fmt.Errorf("flux2: nil model")
	}
	if !webgpu.Available() {
		if err := webgpu.InitError(); err != nil {
			return err
		}
		return fmt.Errorf("webgpu: no adapter")
	}
	webgpu.ClearBinaryWeightCache()

	blobs := m.collectBinaryBlobs()
	sort.Slice(blobs, func(i, j int) bool {
		return len(blobs[i].Raw) > len(blobs[j].Raw)
	})

	warmed := 0
	for _, b := range blobs {
		if err := warmFluxBinaryBlob(b); err != nil {
			if webgpu.IsBinaryVRAMFull(err) {
				break
			}
			continue
		}
		warmed++
	}
	mb := float64(webgpu.BinaryWeightCacheBytes()) / (1024 * 1024)
	fmt.Printf("  flux2 GPU: resident BinaryG128 %d tensors (%.0f MiB)\n", warmed, mb)
	if warmed == 0 {
		m.UseGPU = false
		m.setLinearsGPU(false)
		return fmt.Errorf("flux2 SyncGPU: could not pin any BinaryG128 in VRAM")
	}
	if warmed < len(blobs) {
		fmt.Printf("  note: %d/%d BinaryG128 on host (VRAM/size)\n", len(blobs)-warmed, len(blobs))
	}
	m.UseGPU = true
	m.setLinearsGPU(true)
	return nil
}

// CloseGPU drops sticky BinaryG128 weights and disables GPU GEMV.
func (m *Model) CloseGPU() {
	if m == nil {
		return
	}
	webgpu.ClearBinaryWeightCache()
	m.UseGPU = false
	m.setLinearsGPU(false)
}

func (m *Model) setLinearsGPU(on bool) {
	for _, l := range m.allLinears() {
		if l != nil && l.Blob != nil && quant.IsBinaryG128(l.Blob) {
			l.UseGPU = on && webgpu.HasBinaryWeight(webgpu.BlobKey(unsafe.Pointer(l.Blob)))
		}
	}
}

func (m *Model) collectBinaryBlobs() []*quant.Blob {
	seen := make(map[uintptr]struct{})
	var out []*quant.Blob
	add := func(b *quant.Blob) {
		if b == nil || !quant.IsBinaryG128(b) || len(b.Raw) < minBinaryGPUBytes {
			return
		}
		k := uintptr(unsafe.Pointer(b))
		if _, ok := seen[k]; ok {
			return
		}
		seen[k] = struct{}{}
		out = append(out, b)
	}
	for _, l := range m.allLinears() {
		if l != nil {
			add(l.Blob)
		}
	}
	return out
}

func (m *Model) allLinears() []*Linear {
	if m == nil {
		return nil
	}
	out := []*Linear{
		m.XEmbedder, m.ContextEmbedder,
		m.TimeLinear1, m.TimeLinear2,
		m.DoubleModImg, m.DoubleModTxt, m.SingleMod,
		m.NormOutLinear, m.ProjOut,
	}
	for i := range m.DoubleBlocks {
		b := &m.DoubleBlocks[i]
		out = append(out,
			b.ToQ, b.ToK, b.ToV, b.AddQ, b.AddK, b.AddV,
			b.ToOut, b.ToAddOut,
			b.FFIn, b.FFOut, b.FFContextIn, b.FFContextOut,
		)
	}
	for i := range m.SingleBlocks {
		b := &m.SingleBlocks[i]
		out = append(out, b.ToQKVMLP, b.ToOut)
	}
	return out
}

func warmFluxBinaryBlob(b *quant.Blob) error {
	scales, words, g128, ok := dense.BinaryBlobStaging(b)
	if !ok {
		return fmt.Errorf("binary staging failed")
	}
	return webgpu.WarmBinaryWeight(webgpu.BlobKey(unsafe.Pointer(b)), scales, words, b.Rows, b.Cols, g128)
}
