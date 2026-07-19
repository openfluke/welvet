package convt2

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
)

type Layer struct {
	Core core.Layer
	Cfg  Config
	Exec core.ExecConfig
	Proj *dense.Layer
}

func New(cfg Config) (*Layer, error) {
	return NewConfigured[float64](cfg, core.DTypeFloat32, quant.FormatNone, nil)
}

func NewConfigured[T core.Numeric](cfg Config, dt core.DType, format quant.Format, init []T) (*Layer, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	proj, err := dense.NewConfigured(cfg.PatchDim(), cfg.Filters, cfg.Activation, dt, format, init)
	if err != nil {
		return nil, fmt.Errorf("convt2 proj: %w", err)
	}
	l := &Layer{
		Core: core.Layer{
			Type:         core.LayerConvTransposed2D,
			DType:        dt,
			Activation:   cfg.Activation,
			InputHeight:  cfg.Height,
			OutputHeight: cfg.OutH(),
			TileSize:     32,
			MultiCore:    true,
		},
		Cfg:  cfg,
		Proj: proj,
		Exec: core.ExecConfig{
			Backend:   core.BackendCPUTiled,
			MultiCore: true,
			TileSize:  32,
		},
	}
	l.syncProjExec()
	return l, nil
}

func (l *Layer) syncProjExec() {
	if l == nil || l.Proj == nil {
		return
	}
	l.Proj.Exec = l.Exec
	l.Proj.Core.TileSize = l.Core.TileSize
	l.Proj.Core.MultiCore = l.Core.MultiCore
	l.Proj.Core.Activation = l.Cfg.Activation
	l.Core.Activation = l.Cfg.Activation
}

func (l *Layer) SetDType(dt core.DType) error {
	if l == nil || l.Proj == nil {
		return fmt.Errorf("convt2: nil")
	}
	if err := l.Proj.Weights.SetDType(dt); err != nil {
		return err
	}
	l.Core.DType = dt
	l.Proj.Core.DType = dt
	return nil
}

func (l *Layer) Pack(format quant.Format) error {
	if l == nil || l.Proj == nil {
		return fmt.Errorf("convt2: nil")
	}
	return l.Proj.Weights.Pack(format)
}

func Forward[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if l == nil || l.Proj == nil || input == nil {
		return nil, nil, fmt.Errorf("convt2: nil layer/input")
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

func Backward[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if l == nil || l.Proj == nil {
		return nil, nil, fmt.Errorf("convt2: nil layer")
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

func (l *Layer) GradWSize() int {
	if l == nil || l.Proj == nil || l.Proj.Weights == nil {
		return 0
	}
	return l.Proj.Weights.Rows * l.Proj.Weights.Cols
}

func PermutationOK(dt core.DType, format quant.Format, backend core.Backend) bool {
	return dense.PermutationOK(dt, format, backend)
}

func AllPermutations() (out []struct {
	DType   core.DType
	Format  quant.Format
	Backend core.Backend
}) {
	return dense.AllPermutations()
}
