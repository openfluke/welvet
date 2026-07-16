package rmsnorm

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/dense"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/weights"
)

// Layer is RMSNorm with learnable γ (no bias).
type Layer struct {
	Core  core.Layer
	Cfg   Config
	Exec  core.ExecConfig
	Gamma *weights.Store // shape 1×Dim
}

// New creates RMSNorm with γ=1 FormatNone.
func New(cfg Config) (*Layer, error) {
	ones := make([]float64, cfg.Dim)
	for i := range ones {
		ones[i] = 1
	}
	return NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone, ones)
}

// NewConfigured builds RMSNorm from γ init (length Dim).
func NewConfigured[T core.Numeric](cfg Config, dt core.DType, format quant.Format, gamma []T) (*Layer, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	ws, err := weights.New(1, cfg.Dim, gamma, dt, format)
	if err != nil {
		return nil, fmt.Errorf("rmsnorm gamma: %w", err)
	}
	return &Layer{
		Core: core.Layer{
			Type:         core.LayerRMSNorm,
			DType:        dt,
			Activation:   core.ActivationLinear,
			InputHeight:  cfg.Dim,
			OutputHeight: cfg.Dim,
			TileSize:     32,
			MultiCore:    true,
		},
		Cfg:   cfg,
		Gamma: ws,
		Exec: core.ExecConfig{
			Backend:   core.BackendCPUTiled,
			MultiCore: true,
			TileSize:  32,
		},
	}, nil
}

// SetDType sets γ dtype.
func (l *Layer) SetDType(dt core.DType) error {
	if l == nil || l.Gamma == nil {
		return fmt.Errorf("rmsnorm: nil")
	}
	if err := l.Gamma.SetDType(dt); err != nil {
		return err
	}
	l.Core.DType = dt
	return nil
}

// Pack packs γ to format.
func (l *Layer) Pack(format quant.Format) error {
	if l == nil || l.Gamma == nil {
		return fmt.Errorf("rmsnorm: nil")
	}
	return l.Gamma.Pack(format)
}

// Forward dispatches by Exec.Backend.
func Forward[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if l == nil || l.Gamma == nil || input == nil {
		return nil, nil, fmt.Errorf("rmsnorm: nil layer/input")
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
// pre must be x̂ = x/rms from Forward (before γ).
func Backward[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if l == nil || l.Gamma == nil {
		return nil, nil, fmt.Errorf("rmsnorm: nil layer")
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

// GradWSize is len(γ).
func (l *Layer) GradWSize() int {
	if l == nil || l.Gamma == nil {
		return 0
	}
	return l.Gamma.Rows * l.Gamma.Cols
}

// PermutationOK — same coverage as Dense weight cells (γ store).
func PermutationOK(dt core.DType, format quant.Format, backend core.Backend) bool {
	return dense.PermutationOK(dt, format, backend)
}

// AllPermutations lists the RMSNorm coverage matrix.
func AllPermutations() (out []struct {
	DType   core.DType
	Format  quant.Format
	Backend core.Backend
}) {
	return dense.AllPermutations()
}
