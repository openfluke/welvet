package rnn

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
)

// Layer is a vanilla RNN cell. IH / HH are Dense projections (Linear).
type Layer struct {
	Core core.Layer
	Cfg  Config
	Exec core.ExecConfig
	IH   *dense.Layer // Hidden × Input (+ bias)
	HH   *dense.Layer // Hidden × Hidden (no bias)
}

// New creates RNN with zero FormatNone weights and zero bias.
func New(cfg Config) (*Layer, error) {
	return NewConfigured[float64](cfg, core.DTypeFloat32, quant.FormatNone, nil)
}

// NewConfigured builds RNN from optional loom-packed init [ih | hh | bias].
func NewConfigured[T core.Numeric](cfg Config, dt core.DType, format quant.Format, packed []T) (*Layer, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	in, h := cfg.InputSize, cfg.HiddenSize
	ihN, hhN := h*in, h*h
	var ihInit, hhInit []T
	var bias []float64
	if packed != nil {
		want := cfg.WeightCount()
		if len(packed) < want {
			return nil, fmt.Errorf("rnn: packed len %d < %d", len(packed), want)
		}
		ihInit = packed[:ihN]
		hhInit = packed[ihN : ihN+hhN]
		bias = make([]float64, h)
		for i := 0; i < h; i++ {
			bias[i] = core.AsFloat64(packed[ihN+hhN+i])
		}
	}
	ih, err := dense.NewConfigured(in, h, core.ActivationLinear, dt, format, ihInit)
	if err != nil {
		return nil, fmt.Errorf("rnn IH: %w", err)
	}
	hh, err := dense.NewConfigured(h, h, core.ActivationLinear, dt, format, hhInit)
	if err != nil {
		return nil, fmt.Errorf("rnn HH: %w", err)
	}
	if bias != nil {
		ih.Weights.Bias = bias
	} else {
		ih.Weights.Bias = make([]float64, h)
	}
	l := &Layer{
		Core: core.Layer{
			Type:         core.LayerRNN,
			DType:        dt,
			Activation:   core.ActivationTanh,
			InputHeight:  in,
			OutputHeight: h,
			TileSize:     32,
			MultiCore:    true,
		},
		Cfg:  cfg,
		IH:   ih,
		HH:   hh,
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
	if l == nil {
		return
	}
	for _, p := range []*dense.Layer{l.IH, l.HH} {
		if p != nil {
			p.Exec = l.Exec
			p.Core.TileSize = l.Core.TileSize
			p.Core.MultiCore = l.Core.MultiCore
			p.Core.Activation = core.ActivationLinear
		}
	}
}

// SetDType sets IH/HH dtype.
func (l *Layer) SetDType(dt core.DType) error {
	if l == nil || l.IH == nil || l.HH == nil {
		return fmt.Errorf("rnn: nil")
	}
	if err := l.IH.Weights.SetDType(dt); err != nil {
		return err
	}
	if err := l.HH.Weights.SetDType(dt); err != nil {
		return err
	}
	l.Core.DType = dt
	l.IH.Core.DType = dt
	l.HH.Core.DType = dt
	return nil
}

// Pack packs IH and HH.
func (l *Layer) Pack(format quant.Format) error {
	if l == nil || l.IH == nil || l.HH == nil {
		return fmt.Errorf("rnn: nil")
	}
	if err := l.IH.Weights.Pack(format); err != nil {
		return err
	}
	return l.HH.Weights.Pack(format)
}

// Forward dispatches by Exec.Backend.
func Forward[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if l == nil || l.IH == nil || l.HH == nil || input == nil {
		return nil, nil, fmt.Errorf("rnn: nil layer/input")
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

// Backward dispatches by Exec.Backend (BPTT).
// pre must be linear pre-activations (before tanh) from Forward.
// gradW is loom-packed [dIH | dHH | dBias].
func Backward[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if l == nil || l.IH == nil || l.HH == nil {
		return nil, nil, fmt.Errorf("rnn: nil layer")
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

// GradWSize is len(ih)+len(hh)+len(bias).
func (l *Layer) GradWSize() int {
	if l == nil {
		return 0
	}
	return l.Cfg.WeightCount()
}

// PermutationOK — same coverage as Dense.
func PermutationOK(dt core.DType, format quant.Format, backend core.Backend) bool {
	return dense.PermutationOK(dt, format, backend)
}

// AllPermutations lists the RNN coverage matrix.
func AllPermutations() (out []struct {
	DType   core.DType
	Format  quant.Format
	Backend core.Backend
}) {
	return dense.AllPermutations()
}
