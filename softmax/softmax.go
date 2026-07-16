package softmax

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/dense"
	"github.com/openfluke/welvet/quant"
)

// Layer is weightless Softmax.
type Layer struct {
	Core core.Layer
	Cfg  Config
	Exec core.ExecConfig
}

// New creates Softmax with CPU tiled defaults.
func New(cfg Config) (*Layer, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &Layer{
		Core: core.Layer{
			Type:         core.LayerSoftmax,
			DType:        core.DTypeFloat32,
			Activation:   core.ActivationLinear,
			InputHeight:  cfg.Dim,
			OutputHeight: cfg.Dim,
			TileSize:     32,
			MultiCore:    true,
		},
		Cfg: cfg,
		Exec: core.ExecConfig{
			Backend:   core.BackendCPUTiled,
			MultiCore: true,
			TileSize:  32,
		},
	}, nil
}

// SetDType is a no-op for the weightless layer (records Core.DType for harness).
func (l *Layer) SetDType(dt core.DType) error {
	if l == nil {
		return fmt.Errorf("softmax: nil")
	}
	l.Core.DType = dt
	return nil
}

// Pack is a no-op (no weight store).
func (l *Layer) Pack(format quant.Format) error {
	if l == nil {
		return fmt.Errorf("softmax: nil")
	}
	_ = format
	return nil
}

// Forward dispatches by Exec.Backend.
// pre and post are both probabilities y.
func Forward[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if l == nil || input == nil {
		return nil, nil, fmt.Errorf("softmax: nil layer/input")
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

// Backward: dx = (y/T) ⊙ (dy − ⟨dy,y⟩); gradW is always nil.
// pre must be probabilities from Forward.
func Backward[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if l == nil {
		return nil, nil, fmt.Errorf("softmax: nil layer")
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

// GradWSize is 0.
func (l *Layer) GradWSize() int { return 0 }

// PermutationOK — ALU runs for every Dense coverage cell (weightless).
func PermutationOK(dt core.DType, format quant.Format, backend core.Backend) bool {
	return dense.PermutationOK(dt, format, backend)
}

// AllPermutations lists the harness coverage matrix (ALU-only cells).
func AllPermutations() (out []struct {
	DType   core.DType
	Format  quant.Format
	Backend core.Backend
}) {
	return dense.AllPermutations()
}
