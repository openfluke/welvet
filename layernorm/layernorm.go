package layernorm

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/dense"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/weights"
)

// Layer is LayerNorm with learnable γ and β.
type Layer struct {
	Core  core.Layer
	Cfg   Config
	Exec  core.ExecConfig
	Gamma *weights.Store // shape 1×Dim
	Beta  *weights.Store // shape 1×Dim
}

// New creates LayerNorm with γ=1, β=0 FormatNone.
func New(cfg Config) (*Layer, error) {
	ones := make([]float64, cfg.Dim)
	zeros := make([]float64, cfg.Dim)
	for i := range ones {
		ones[i] = 1
	}
	return NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone, ones, zeros)
}

// NewConfigured builds LayerNorm from γ,β init (each length Dim).
func NewConfigured[T core.Numeric](cfg Config, dt core.DType, format quant.Format, gamma, beta []T) (*Layer, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	gs, err := weights.New(1, cfg.Dim, gamma, dt, format)
	if err != nil {
		return nil, fmt.Errorf("layernorm gamma: %w", err)
	}
	bs, err := weights.New(1, cfg.Dim, beta, dt, format)
	if err != nil {
		return nil, fmt.Errorf("layernorm beta: %w", err)
	}
	return &Layer{
		Core: core.Layer{
			Type:         core.LayerLayerNorm,
			DType:        dt,
			Activation:   core.ActivationLinear,
			InputHeight:  cfg.Dim,
			OutputHeight: cfg.Dim,
			TileSize:     32,
			MultiCore:    true,
		},
		Cfg:   cfg,
		Gamma: gs,
		Beta:  bs,
		Exec: core.ExecConfig{
			Backend:   core.BackendCPUTiled,
			MultiCore: true,
			TileSize:  32,
		},
	}, nil
}

// SetDType sets γ and β dtype.
func (l *Layer) SetDType(dt core.DType) error {
	if l == nil || l.Gamma == nil || l.Beta == nil {
		return fmt.Errorf("layernorm: nil")
	}
	if err := l.Gamma.SetDType(dt); err != nil {
		return err
	}
	if err := l.Beta.SetDType(dt); err != nil {
		return err
	}
	l.Core.DType = dt
	return nil
}

// Pack packs γ and β to format.
func (l *Layer) Pack(format quant.Format) error {
	if l == nil || l.Gamma == nil || l.Beta == nil {
		return fmt.Errorf("layernorm: nil")
	}
	if err := l.Gamma.Pack(format); err != nil {
		return err
	}
	return l.Beta.Pack(format)
}

// Forward dispatches by Exec.Backend.
func Forward[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if l == nil || l.Gamma == nil || l.Beta == nil || input == nil {
		return nil, nil, fmt.Errorf("layernorm: nil layer/input")
	}
	switch l.Exec.Backend {
	case core.BackendSIMD:
		return ForwardSIMD(l, input)
	case core.BackendWebGPU:
		return ForwardWebGPU(l, input)
	default:
		return ForwardCPUTiled(l, input)
	}
}

// Backward dispatches by Exec.Backend.
// pre must be x̂ = (x−μ)/σ from Forward (before affine).
// gradW is [dγ | dβ] length 2·Dim.
func Backward[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if l == nil || l.Gamma == nil || l.Beta == nil {
		return nil, nil, fmt.Errorf("layernorm: nil layer")
	}
	switch l.Exec.Backend {
	case core.BackendSIMD:
		return BackwardSIMD(l, gradOut, input, pre)
	case core.BackendWebGPU:
		return BackwardWebGPU(l, gradOut, input, pre)
	default:
		return BackwardCPUTiled(l, gradOut, input, pre)
	}
}

// GradWSize is len(γ)+len(β).
func (l *Layer) GradWSize() int {
	if l == nil || l.Gamma == nil || l.Beta == nil {
		return 0
	}
	return l.Gamma.Rows*l.Gamma.Cols + l.Beta.Rows*l.Beta.Cols
}

// PermutationOK — same coverage as Dense weight cells (γ/β stores).
func PermutationOK(dt core.DType, format quant.Format, backend core.Backend) bool {
	return dense.PermutationOK(dt, format, backend)
}

// AllPermutations lists the LayerNorm coverage matrix.
func AllPermutations() (out []struct {
	DType   core.DType
	Format  quant.Format
	Backend core.Backend
}) {
	return dense.AllPermutations()
}
