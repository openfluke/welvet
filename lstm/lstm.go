package lstm

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/dense"
	"github.com/openfluke/welvet/quant"
)

// Gate is one LSTM gate (IH + HH Dense, bias on IH).
type Gate struct {
	IH *dense.Layer
	HH *dense.Layer
}

// Layer is an LSTM cell with gates i, f, g, o.
type Layer struct {
	Core     core.Layer
	Cfg      Config
	Exec     core.ExecConfig
	I, F, G, O *Gate
}

// New creates LSTM with zero FormatNone weights.
func New(cfg Config) (*Layer, error) {
	return NewConfigured[float64](cfg, core.DTypeFloat32, quant.FormatNone, nil)
}

// NewConfigured builds LSTM from optional loom-packed init [i|f|g|o] gates.
func NewConfigured[T core.Numeric](cfg Config, dt core.DType, format quant.Format, packed []T) (*Layer, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	gateN := cfg.GateSize()
	if packed != nil && len(packed) < cfg.WeightCount() {
		return nil, fmt.Errorf("lstm: packed len %d < %d", len(packed), cfg.WeightCount())
	}
	mk := func(name string, off int) (*Gate, error) {
		var slice []T
		if packed != nil {
			slice = packed[off : off+gateN]
		}
		g, err := newGate(cfg, dt, format, slice)
		if err != nil {
			return nil, fmt.Errorf("lstm %s: %w", name, err)
		}
		return g, nil
	}
	gi, err := mk("i", 0)
	if err != nil {
		return nil, err
	}
	gf, err := mk("f", gateN)
	if err != nil {
		return nil, err
	}
	gg, err := mk("g", 2*gateN)
	if err != nil {
		return nil, err
	}
	go_, err := mk("o", 3*gateN)
	if err != nil {
		return nil, err
	}
	l := &Layer{
		Core: core.Layer{
			Type:         core.LayerLSTM,
			DType:        dt,
			Activation:   core.ActivationTanh,
			InputHeight:  cfg.InputSize,
			OutputHeight: cfg.HiddenSize,
			TileSize:     32,
			MultiCore:    true,
		},
		Cfg:  cfg,
		I:    gi,
		F:    gf,
		G:    gg,
		O:    go_,
		Exec: core.ExecConfig{
			Backend:   core.BackendCPUTiled,
			MultiCore: true,
			TileSize:  32,
		},
	}
	l.syncProjExec()
	return l, nil
}

func newGate[T core.Numeric](cfg Config, dt core.DType, format quant.Format, packed []T) (*Gate, error) {
	in, h := cfg.InputSize, cfg.HiddenSize
	ihN, hhN := h*in, h*h
	var ihInit, hhInit []T
	var bias []float64
	if packed != nil {
		ihInit = packed[:ihN]
		hhInit = packed[ihN : ihN+hhN]
		bias = make([]float64, h)
		for i := 0; i < h; i++ {
			bias[i] = core.AsFloat64(packed[ihN+hhN+i])
		}
	}
	ih, err := dense.NewConfigured(in, h, core.ActivationLinear, dt, format, ihInit)
	if err != nil {
		return nil, err
	}
	hh, err := dense.NewConfigured(h, h, core.ActivationLinear, dt, format, hhInit)
	if err != nil {
		return nil, err
	}
	if bias != nil {
		ih.Weights.Bias = bias
	} else {
		ih.Weights.Bias = make([]float64, h)
	}
	return &Gate{IH: ih, HH: hh}, nil
}

func (l *Layer) gates() []*Gate {
	return []*Gate{l.I, l.F, l.G, l.O}
}

func (l *Layer) syncProjExec() {
	if l == nil {
		return
	}
	for _, g := range l.gates() {
		if g == nil {
			continue
		}
		for _, p := range []*dense.Layer{g.IH, g.HH} {
			if p != nil {
				p.Exec = l.Exec
				p.Core.TileSize = l.Core.TileSize
				p.Core.MultiCore = l.Core.MultiCore
				p.Core.Activation = core.ActivationLinear
			}
		}
	}
}

// SetDType sets all gate dtypes.
func (l *Layer) SetDType(dt core.DType) error {
	if l == nil {
		return fmt.Errorf("lstm: nil")
	}
	for _, g := range l.gates() {
		if err := g.IH.Weights.SetDType(dt); err != nil {
			return err
		}
		if err := g.HH.Weights.SetDType(dt); err != nil {
			return err
		}
		g.IH.Core.DType = dt
		g.HH.Core.DType = dt
	}
	l.Core.DType = dt
	return nil
}

// Pack packs all gate weights.
func (l *Layer) Pack(format quant.Format) error {
	if l == nil {
		return fmt.Errorf("lstm: nil")
	}
	for _, g := range l.gates() {
		if err := g.IH.Weights.Pack(format); err != nil {
			return err
		}
		if err := g.HH.Weights.Pack(format); err != nil {
			return err
		}
	}
	return nil
}

// Forward dispatches by Exec.Backend.
func Forward[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if l == nil || l.I == nil || input == nil {
		return nil, nil, fmt.Errorf("lstm: nil layer/input")
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

// Backward dispatches BPTT.
// pre is [batch,seq,5·H] = [i,f,g,o,c] sums/cell; gradW is loom [i|f|g|o] pack.
func Backward[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if l == nil || l.I == nil {
		return nil, nil, fmt.Errorf("lstm: nil layer")
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

// GradWSize is 4 × gate pack.
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

// AllPermutations lists the LSTM coverage matrix.
func AllPermutations() (out []struct {
	DType   core.DType
	Format  quant.Format
	Backend core.Backend
}) {
	return dense.AllPermutations()
}
