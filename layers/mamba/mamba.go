package mamba

import (
	"fmt"
	"math"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/seqmix"
	"github.com/openfluke/welvet/quant"
)

// Layer is a minimal selective SSM (seqmix.KindSSM).
type Layer struct {
	Core   core.Layer
	Cfg    Config
	Exec   core.ExecConfig
	InProj *dense.Layer // DModel → 2*Inner (x | dt)
	OutProj *dense.Layer // Inner → DModel
	ALog   []float32     // [Inner]
	DSkip  []float32     // [Inner]
}

// Kind returns seqmix.KindSSM.
func (l *Layer) Kind() seqmix.Kind { return seqmix.KindSSM }

// New creates Mamba with FormatNone Float32.
func New(cfg Config) (*Layer, error) {
	return NewConfigured[float32](cfg, core.DTypeFloat32, quant.FormatNone, nil, nil, nil, nil)
}

// NewConfigured builds projections. inInit [2*Inner×D], outInit [D×Inner], aLog/dSkip length Inner.
func NewConfigured[T core.Numeric](cfg Config, dt core.DType, format quant.Format, inInit, outInit []T, aLog, dSkip []float32) (*Layer, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	inner := cfg.InnerDim()
	inP, err := dense.NewConfigured(cfg.DModel, 2*inner, core.ActivationLinear, dt, format, inInit)
	if err != nil {
		return nil, fmt.Errorf("mamba in: %w", err)
	}
	outP, err := dense.NewConfigured(inner, cfg.DModel, core.ActivationLinear, dt, format, outInit)
	if err != nil {
		return nil, fmt.Errorf("mamba out: %w", err)
	}
	if aLog == nil {
		aLog = make([]float32, inner)
		for i := range aLog {
			aLog[i] = -float32(i+1) * 0.1
		}
	}
	if dSkip == nil {
		dSkip = make([]float32, inner)
		for i := range dSkip {
			dSkip[i] = 1
		}
	}
	if len(aLog) != inner || len(dSkip) != inner {
		return nil, fmt.Errorf("mamba: ALog/DSkip len want %d", inner)
	}
	l := &Layer{
		Core: core.Layer{
			Type:         core.LayerMamba,
			DType:        dt,
			Activation:   core.ActivationLinear,
			InputHeight:  cfg.DModel,
			OutputHeight: cfg.DModel,
			TileSize:     32,
			MultiCore:    true,
		},
		Cfg:     cfg,
		InProj:  inP,
		OutProj: outP,
		ALog:    append([]float32(nil), aLog...),
		DSkip:   append([]float32(nil), dSkip...),
		Exec: core.ExecConfig{
			Backend:   core.BackendCPUTiled,
			MultiCore: true,
			TileSize:  32,
		},
	}
	l.syncExec()
	return l, nil
}

func (l *Layer) syncExec() {
	if l == nil {
		return
	}
	for _, p := range []*dense.Layer{l.InProj, l.OutProj} {
		if p == nil {
			continue
		}
		p.Exec = l.Exec
		p.Core.TileSize = l.Core.TileSize
		p.Core.MultiCore = l.Core.MultiCore
	}
}

func (l *Layer) SetDType(dt core.DType) error {
	if l == nil {
		return fmt.Errorf("mamba: nil")
	}
	for _, p := range []*dense.Layer{l.InProj, l.OutProj} {
		if err := p.Weights.SetDType(dt); err != nil {
			return err
		}
		p.Core.DType = dt
	}
	l.Core.DType = dt
	return nil
}

func (l *Layer) Pack(format quant.Format) error {
	if l == nil {
		return fmt.Errorf("mamba: nil")
	}
	for _, p := range []*dense.Layer{l.InProj, l.OutProj} {
		if err := p.Weights.Pack(format); err != nil {
			return err
		}
	}
	return nil
}

func Forward[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if l == nil || input == nil {
		return nil, nil, fmt.Errorf("mamba: nil layer/input")
	}
	l.syncExec()
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
	if l == nil {
		return nil, nil, fmt.Errorf("mamba: nil layer")
	}
	l.syncExec()
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
	if l == nil {
		return 0
	}
	n := 0
	for _, p := range []*dense.Layer{l.InProj, l.OutProj} {
		if p != nil && p.Weights != nil {
			n += p.Weights.Rows * p.Weights.Cols
		}
	}
	n += len(l.ALog) + len(l.DSkip)
	return n
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

func softplus(x float64) float64 {
	if x > 20 {
		return x
	}
	return math.Log1p(math.Exp(x))
}
