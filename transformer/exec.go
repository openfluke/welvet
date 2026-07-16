package transformer

import (
	"fmt"

	"github.com/openfluke/welvet/core"
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
		return "packed quant + SIMD fused kernels (Lucy-style CPU path)"
	case "gpu_fuse":
		if webgpu.Available() {
			return "packed quant dense + LM head on GPU; embed/norm/attn on host"
		}
		return "packed quant on GPU (needs Vulkan/DX12/Metal adapter)"
	default:
		return ""
	}
}

// GPUHybridNote explains what runs on GPU vs host for the gpu profile.
func (p ExecProfile) GPUHybridNote() string {
	if p.Backend != core.BackendWebGPU {
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
	}
	return nil
}
