package swiglu

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
)

// Layer is SwiGLU. Gate/Up/Down are dense.Layer units.
type Layer struct {
	Core core.Layer
	Cfg  Config
	Exec core.ExecConfig

	Gate, Up, Down *dense.Layer
}

// New creates SwiGLU with zero FormatNone weights.
func New(cfg Config) (*Layer, error) {
	return NewConfigured[float64](cfg, core.DTypeFloat32, quant.FormatNone, nil, nil, nil)
}

// NewConfigured builds SwiGLU with optional init (row-major [out×in] each).
func NewConfigured[T core.Numeric](
	cfg Config,
	dt core.DType,
	format quant.Format,
	gateInit, upInit, downInit []T,
) (*Layer, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	in, inter := cfg.InputDim, cfg.IntermediateDim

	Gate, err := dense.NewConfigured(in, inter, core.ActivationLinear, dt, format, gateInit)
	if err != nil {
		return nil, fmt.Errorf("swiglu Gate: %w", err)
	}
	Up, err := dense.NewConfigured(in, inter, core.ActivationLinear, dt, format, upInit)
	if err != nil {
		return nil, fmt.Errorf("swiglu Up: %w", err)
	}
	Down, err := dense.NewConfigured(inter, in, core.ActivationLinear, dt, format, downInit)
	if err != nil {
		return nil, fmt.Errorf("swiglu Down: %w", err)
	}

	l := &Layer{
		Core: core.Layer{
			Type:         core.LayerSwiGLU,
			DType:        dt,
			Activation:   core.ActivationSilu,
			InputHeight:  in,
			OutputHeight: in,
			TileSize:     32,
			MultiCore:    true,
		},
		Cfg: cfg,
		Exec: core.ExecConfig{
			Backend:   core.BackendCPUTiled,
			MultiCore: true,
			TileSize:  32,
		},
		Gate: Gate, Up: Up, Down: Down,
	}
	l.syncProjExec()
	return l, nil
}

func (l *Layer) syncProjExec() {
	if l == nil {
		return
	}
	for _, p := range []*dense.Layer{l.Gate, l.Up, l.Down} {
		if p != nil {
			p.Exec = l.Exec
			p.Core.TileSize = l.Core.TileSize
			p.Core.MultiCore = l.Core.MultiCore
		}
	}
}

// SetDType sets weight dtype on Gate/Up/Down.
func (l *Layer) SetDType(dt core.DType) error {
	if l == nil {
		return fmt.Errorf("swiglu: nil")
	}
	for _, p := range []*dense.Layer{l.Gate, l.Up, l.Down} {
		if err := p.Weights.SetDType(dt); err != nil {
			return err
		}
	}
	l.Core.DType = dt
	return nil
}

// Pack packs all three projections.
func (l *Layer) Pack(format quant.Format) error {
	if l == nil {
		return fmt.Errorf("swiglu: nil")
	}
	for _, p := range []*dense.Layer{l.Gate, l.Up, l.Down} {
		if err := p.Weights.Pack(format); err != nil {
			return err
		}
	}
	return nil
}

// Forward dispatches by Exec.Backend.
func Forward[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if l == nil || l.Gate == nil || input == nil {
		return nil, nil, fmt.Errorf("swiglu: nil layer/input")
	}
	l.syncProjExec()
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
// pre must be the gated hidden h from Forward (shape matching IntermediateDim).
func Backward[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if l == nil || l.Gate == nil {
		return nil, nil, fmt.Errorf("swiglu: nil layer")
	}
	l.syncProjExec()
	switch l.Exec.Backend {
	case core.BackendSIMD:
		return BackwardSIMD(l, gradOut, input, pre)
	case core.BackendWebGPU:
		return BackwardWebGPU(l, gradOut, input, pre)
	default:
		return BackwardCPUTiled(l, gradOut, input, pre)
	}
}

// GradWSize is concat(Gate, Up, Down) weight matrix lengths.
func (l *Layer) GradWSize() int {
	if l == nil {
		return 0
	}
	n := 0
	for _, p := range []*dense.Layer{l.Gate, l.Up, l.Down} {
		if p != nil && p.Weights != nil {
			n += p.Weights.Rows * p.Weights.Cols
		}
	}
	return n
}

// PermutationOK — same coverage as Dense projections.
func PermutationOK(dt core.DType, format quant.Format, backend core.Backend) bool {
	return dense.PermutationOK(dt, format, backend)
}

// AllPermutations lists the SwiGLU coverage matrix.
func AllPermutations() (out []struct {
	DType   core.DType
	Format  quant.Format
	Backend core.Backend
}) {
	return dense.AllPermutations()
}
