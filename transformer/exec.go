package transformer

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/dense"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/simd"
	"github.com/openfluke/welvet/webgpu"
)

// ExecProfile is a named host/GPU execution setting for generate.
type ExecProfile struct {
	Name       string
	Backend    core.Backend
	MultiCore  bool
	TileSize   int
	Fused      bool // packed-quant fused matmul (simd_fuse / gpu_fuse)
	PackFormat quant.Format
}

// ProfileSIMDMultiCore is the default Octo/Lucy-style host path.
func ProfileSIMDMultiCore() ExecProfile {
	return ExecProfile{Name: "simd_mc", Backend: core.BackendSIMD, MultiCore: true, TileSize: 32}
}

// NamedProfiles lists selectable run profiles (Enter default = simd_mc).
func NamedProfiles() []ExecProfile {
	return []ExecProfile{
		{Name: "cpu_sc", Backend: core.BackendCPUTiled, MultiCore: false, TileSize: 32},
		{Name: "cpu_mc", Backend: core.BackendCPUTiled, MultiCore: true, TileSize: 32},
		{Name: "simd_sc", Backend: core.BackendSIMD, MultiCore: false, TileSize: 32},
		{Name: "simd_mc", Backend: core.BackendSIMD, MultiCore: true, TileSize: 32},
		{Name: "gpu", Backend: core.BackendWebGPU, MultiCore: true, TileSize: 32},
		{Name: "simd_fuse", Backend: core.BackendSIMD, MultiCore: true, TileSize: 32, Fused: true},
		{Name: "gpu_fuse", Backend: core.BackendWebGPU, MultiCore: true, TileSize: 32, Fused: true},
	}
}

// Validate checks the profile can run on this host.
func (p ExecProfile) Validate() error {
	if p.TileSize < 0 {
		return fmt.Errorf("tile size must be >= 0")
	}
	switch p.Backend {
	case core.BackendCPUTiled:
		return nil
	case core.BackendSIMD:
		if !simd.Enabled() {
			return fmt.Errorf("simd requires Plan 9 AVX2/NEON kernels (GOARCH unsupported or not linked)")
		}
		return nil
	case core.BackendWebGPU:
		if !webgpu.Available() {
			if err := webgpu.InitError(); err != nil {
				return fmt.Errorf("WebGPU: %w (set WELVET_WGPU_BACKEND=vulkan|dx12|metal if needed)", err)
			}
			return fmt.Errorf("WebGPU: no adapter (set WELVET_WGPU_BACKEND=vulkan|dx12|metal)")
		}
		return nil
	default:
		return fmt.Errorf("unknown backend %v", p.Backend)
	}
}

// String describes the profile for logs.
func (p ExecProfile) String() string {
	cores := "sc"
	if p.MultiCore {
		cores = "mc"
	}
	tile := p.TileSize
	if tile <= 0 {
		tile = 32
	}
	s := fmt.Sprintf("%s (%s, %s, tile=%d)", p.Name, p.Backend.String(), cores, tile)
	if p.Fused && p.PackFormat != quant.FormatNone {
		s += fmt.Sprintf(", pack=%s", p.PackFormat.String())
	}
	return s
}

// FusedNote explains fused packing for menu / logs.
func (p ExecProfile) FusedNote() string {
	if !p.Fused {
		return ""
	}
	switch p.Name {
	case "simd_fuse":
		return "packed fused GEMV (Q4/Q8/Q4_1/Q5 asm; k/IQ inflate-once + DotTile)"
	case "gpu_fuse":
		return "full on-device fused decoder (Q4 Lucy, or BinaryG128 hybrid — all weights on GPU; ~8GB+ VRAM for Bonsai)"
	default:
		return ""
	}
}

// GPUHybridNote explains what runs on GPU vs host for the gpu profile.
func (p ExecProfile) GPUHybridNote() string {
	if p.Backend != core.BackendWebGPU {
		return ""
	}
	if p.Fused {
		return ""
	}
	name := webgpu.AdapterName()
	if name != "" {
		return fmt.Sprintf("hybrid GPU: dense Q/K/V/O + MLP + LM head on %s; embed/norm/attn ALU on host", name)
	}
	return "hybrid GPU: dense projections + LM head on device; embed/norm/attn ALU on host"
}

// ApplyExec sets Backend / MultiCore / TileSize on every layer used by generate.
func (m *Model) ApplyExec(p ExecProfile) error {
	if m == nil {
		return fmt.Errorf("transformer: nil model")
	}
	if err := p.Validate(); err != nil {
		return err
	}
	tile := p.TileSize
	if tile <= 0 {
		tile = 32
	}
	exec := core.ExecConfig{
		Backend:   p.Backend,
		MultiCore: p.MultiCore,
		TileSize:  tile,
		UseSIMD:   p.Backend == core.BackendSIMD,
		UseWebGPU: p.Backend == core.BackendWebGPU,
	}
	m.Exec = exec
	m.Fused = p.Fused
	if p.Fused {
		format := p.PackFormat
		if m.FusedPack {
			if format != quant.FormatNone && format != m.PackFormat {
				return fmt.Errorf("entity baked as %s; profile asked for %s (re-convert or pick matching format)",
					m.PackFormat.String(), format.String())
			}
			format = m.PackFormat
			m.Fused = true
		} else {
			if format == quant.FormatNone {
				format = quant.FormatQ4_0
			}
			if err := m.EnsureFused(format); err != nil {
				return fmt.Errorf("fused pack: %w", err)
			}
		}
	}
	if p.Fused && p.Backend == core.BackendWebGPU {
		if m.FusedGPUReady() {
			if err := m.SyncGPU(); err != nil {
				return fmt.Errorf("fused gpu: %w", err)
			}
		} else if m.IsHybrid() {
			m.CloseGPU()
			m.CloseHybridGPU()
			m.Fused = true
			if err := m.SyncHybridFused(); err != nil {
				return fmt.Errorf("hybrid gpu fuse: %w", err)
			}
			name := m.GPUAdapterName()
			vramHint := "needs ~8GB+ VRAM for 27B; ~2–3GB for dense 8B"
			if m.Architecture == "qwen3_dense" {
				vramHint = "dense BinaryG128 (~2–3GB VRAM)"
			}
			if name != "" {
				fmt.Printf("  gpu_fuse (BinaryG128): full on-device fuse on %s (%s)\n", name, vramHint)
			} else {
				fmt.Printf("  gpu_fuse (BinaryG128): full on-device fuse (%s)\n", vramHint)
			}
		} else {
			return fmt.Errorf("gpu_fuse requires a baked packed entity (got %s)", m.PackFormat.String())
		}
	} else {
		m.CloseGPU()
		m.CloseHybridGPU()
		if err := m.ensureHostPackedWeights(); err != nil {
			return err
		}
	}
	if m.Embed != nil {
		m.Embed.Exec = exec
		m.Embed.Core.TileSize = tile
		m.Embed.Core.MultiCore = p.MultiCore
	}
	if m.FinalNorm != nil {
		m.FinalNorm.Exec = exec
		m.FinalNorm.Core.TileSize = tile
		m.FinalNorm.Core.MultiCore = p.MultiCore
	}
	for i := range m.Blocks {
		b := &m.Blocks[i]
		if b.AttnNorm != nil {
			b.AttnNorm.Exec = exec
			b.AttnNorm.Core.TileSize = tile
			b.AttnNorm.Core.MultiCore = p.MultiCore
		}
		if b.FFNNorm != nil {
			b.FFNNorm.Exec = exec
			b.FFNNorm.Core.TileSize = tile
			b.FFNNorm.Core.MultiCore = p.MultiCore
		}
		if b.Attn != nil {
			b.Attn.Exec = exec
			b.Attn.Core.TileSize = tile
			b.Attn.Core.MultiCore = p.MultiCore
			// Projections pick up Exec via syncProjExec on next Forward.
			b.Attn.Q.Exec = exec
			b.Attn.K.Exec = exec
			b.Attn.V.Exec = exec
			b.Attn.O.Exec = exec
			b.Attn.Q.Core.TileSize = tile
			b.Attn.K.Core.TileSize = tile
			b.Attn.V.Core.TileSize = tile
			b.Attn.O.Core.TileSize = tile
			b.Attn.Q.Core.MultiCore = p.MultiCore
			b.Attn.K.Core.MultiCore = p.MultiCore
			b.Attn.V.Core.MultiCore = p.MultiCore
			b.Attn.O.Core.MultiCore = p.MultiCore
		}
		if b.FFN != nil {
			b.FFN.Exec = exec
			b.FFN.Core.TileSize = tile
			b.FFN.Core.MultiCore = p.MultiCore
			b.FFN.Gate.Exec = exec
			b.FFN.Up.Exec = exec
			b.FFN.Down.Exec = exec
			b.FFN.Gate.Core.TileSize = tile
			b.FFN.Up.Core.TileSize = tile
			b.FFN.Down.Core.TileSize = tile
			b.FFN.Gate.Core.MultiCore = p.MultiCore
			b.FFN.Up.Core.MultiCore = p.MultiCore
			b.FFN.Down.Core.MultiCore = p.MultiCore
		}
		// Hybrid Qwen3.5 / Bonsai projections (full-attn layers)
		for _, d := range []*dense.Layer{b.Q, b.K, b.V, b.O} {
			if d == nil {
				continue
			}
			d.Exec = exec
			d.Core.TileSize = tile
			d.Core.MultiCore = p.MultiCore
		}
	}
	return nil
}
