package sequential

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
)

// Layer is Sequential compose of Dense children.
type Layer struct {
	Core     core.Layer
	Cfg      Config
	Exec     core.ExecConfig
	Children []*dense.Layer
}

// New creates Sequential with Depth square Dense children (FormatNone Float32).
func New(cfg Config) (*Layer, error) {
	return NewConfigured[float32](cfg, core.DTypeFloat32, quant.FormatNone, nil)
}

// NewConfigured builds Sequential. packed is optional concat of child [out×in] weights
// (length Depth×Dim×Dim); nil → zeros.
func NewConfigured[T core.Numeric](cfg Config, dt core.DType, format quant.Format, packed []T) (*Layer, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	dim, depth := cfg.Dim, cfg.Depth
	childN := dim * dim
	children := make([]*dense.Layer, depth)
	for i := 0; i < depth; i++ {
		var init []T
		if packed != nil {
			off := i * childN
			if len(packed) < off+childN {
				return nil, fmt.Errorf("sequential: packed short at child %d", i)
			}
			init = packed[off : off+childN]
		}
		ch, err := dense.NewConfigured(dim, dim, core.ActivationLinear, dt, format, init)
		if err != nil {
			return nil, fmt.Errorf("sequential child %d: %w", i, err)
		}
		children[i] = ch
	}
	l := &Layer{
		Core: core.Layer{
			Type:         core.LayerSequential,
			DType:        dt,
			Activation:   core.ActivationLinear,
			InputHeight:  dim,
			OutputHeight: dim,
			TileSize:     32,
			MultiCore:    true,
		},
		Cfg:      cfg,
		Children: children,
		Exec: core.ExecConfig{
			Backend:   core.BackendCPUTiled,
			MultiCore: true,
			TileSize:  32,
		},
	}
	l.syncChildExec()
	return l, nil
}

func (l *Layer) syncChildExec() {
	if l == nil {
		return
	}
	for _, ch := range l.Children {
		if ch == nil {
			continue
		}
		ch.Exec = l.Exec
		ch.Core.TileSize = l.Core.TileSize
		ch.Core.MultiCore = l.Core.MultiCore
	}
}

// SetDType sets dtype on all children.
func (l *Layer) SetDType(dt core.DType) error {
	if l == nil {
		return fmt.Errorf("sequential: nil")
	}
	for i, ch := range l.Children {
		if err := ch.Weights.SetDType(dt); err != nil {
			return fmt.Errorf("sequential child %d: %w", i, err)
		}
	}
	l.Core.DType = dt
	return nil
}

// Pack packs all children.
func (l *Layer) Pack(format quant.Format) error {
	if l == nil {
		return fmt.Errorf("sequential: nil")
	}
	for i, ch := range l.Children {
		if err := ch.Weights.Pack(format); err != nil {
			return fmt.Errorf("sequential child %d: %w", i, err)
		}
	}
	return nil
}

// Forward dispatches by Exec.Backend (propagated to children).
func Forward[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if l == nil || input == nil {
		return nil, nil, fmt.Errorf("sequential: nil layer/input")
	}
	l.syncChildExec()
	switch l.Exec.Backend {
	case core.BackendSIMD:
		return ForwardSIMD(l, input)
	case core.BackendWebGPU:
		return ForwardWebGPU(l, input)
	default:
		return ForwardCPUTiled(l, input)
	}
}

// Backward chains child Dense backwards; gradW is concat of child dWs.
// pre is unused (tape recomputed from input); may be nil.
func Backward[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if l == nil {
		return nil, nil, fmt.Errorf("sequential: nil layer")
	}
	l.syncChildExec()
	switch l.Exec.Backend {
	case core.BackendSIMD:
		return BackwardSIMD(l, gradOut, input, pre)
	case core.BackendWebGPU:
		return BackwardWebGPU(l, gradOut, input, pre)
	default:
		return BackwardCPUTiled(l, gradOut, input, pre)
	}
}

// GradWSize is sum of child weight matrix lengths.
func (l *Layer) GradWSize() int {
	if l == nil {
		return 0
	}
	n := 0
	for _, ch := range l.Children {
		if ch != nil && ch.Weights != nil {
			n += ch.Weights.Rows * ch.Weights.Cols
		}
	}
	return n
}

// PermutationOK — same coverage as Dense children.
func PermutationOK(dt core.DType, format quant.Format, backend core.Backend) bool {
	return dense.PermutationOK(dt, format, backend)
}

// AllPermutations lists the Sequential coverage matrix.
func AllPermutations() (out []struct {
	DType   core.DType
	Format  quant.Format
	Backend core.Backend
}) {
	return dense.AllPermutations()
}
