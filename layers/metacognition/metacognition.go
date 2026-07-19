package metacognition

import (
	"fmt"
	"math"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
)

// Stats summarizes a tensor for heuristics.
type Stats struct {
	Avg, Std, Max, Min float64
	Active, Total      int
}

// Layer wraps Observed Dense with heuristic rules.
type Layer struct {
	Core     core.Layer
	Cfg      Config
	Exec     core.ExecConfig
	Observed *dense.Layer
	Rules    []Rule
	lastGate float64 // 1 = full, <1 gated
}

// New creates Metacognition with a square Dense Observed.
func New(cfg Config) (*Layer, error) {
	return NewConfigured[float32](cfg, core.DTypeFloat32, quant.FormatNone, nil)
}

// NewConfigured builds Observed Dim→Dim Dense.
func NewConfigured[T core.Numeric](cfg Config, dt core.DType, format quant.Format, init []T) (*Layer, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	obs, err := dense.NewConfigured(cfg.Dim, cfg.Dim, core.ActivationLinear, dt, format, init)
	if err != nil {
		return nil, fmt.Errorf("metacognition observed: %w", err)
	}
	rules := append([]Rule(nil), cfg.Rules...)
	if len(rules) == 0 {
		rules = DefaultStabilityRules()
	}
	l := &Layer{
		Core: core.Layer{
			Type:         core.LayerMetacognition,
			DType:        dt,
			Activation:   core.ActivationLinear,
			InputHeight:  cfg.Dim,
			OutputHeight: cfg.Dim,
			TileSize:     32,
			MultiCore:    true,
		},
		Cfg:      cfg,
		Observed: obs,
		Rules:    rules,
		lastGate: 1,
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
	if l == nil || l.Observed == nil {
		return
	}
	l.Observed.Exec = l.Exec
	l.Observed.Core.TileSize = l.Core.TileSize
	l.Observed.Core.MultiCore = l.Core.MultiCore
}

func (l *Layer) SetDType(dt core.DType) error {
	if l == nil || l.Observed == nil {
		return fmt.Errorf("metacognition: nil")
	}
	if err := l.Observed.Weights.SetDType(dt); err != nil {
		return err
	}
	l.Core.DType = dt
	l.Observed.Core.DType = dt
	return nil
}

func (l *Layer) Pack(format quant.Format) error {
	if l == nil || l.Observed == nil {
		return fmt.Errorf("metacognition: nil")
	}
	return l.Observed.Weights.Pack(format)
}

func Forward[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if l == nil || input == nil {
		return nil, nil, fmt.Errorf("metacognition: nil layer/input")
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
		return nil, nil, fmt.Errorf("metacognition: nil layer")
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
	if l == nil || l.Observed == nil || l.Observed.Weights == nil {
		return 0
	}
	return l.Observed.Weights.Rows * l.Observed.Weights.Cols
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

func computeStats[T core.Numeric](t *core.Tensor[T]) Stats {
	if t == nil || t.Len() == 0 {
		return Stats{}
	}
	n := t.Len()
	var sum, sumSq, mx, mn float64
	mx = core.AsFloat64(t.Data[0])
	mn = mx
	active := 0
	for i := 0; i < n; i++ {
		v := core.AsFloat64(t.Data[i])
		sum += v
		sumSq += v * v
		if v > mx {
			mx = v
		}
		if v < mn {
			mn = v
		}
		if math.Abs(v) > 1e-6 {
			active++
		}
	}
	avg := sum / float64(n)
	var std float64
	if n > 1 {
		std = math.Sqrt(math.Max(0, sumSq/float64(n)-avg*avg))
	}
	return Stats{Avg: avg, Std: std, Max: mx, Min: mn, Active: active, Total: n}
}

func evalRule(r *Rule, inS, outS Stats) bool {
	switch r.Condition {
	case CondStdAbove:
		return outS.Std > r.Threshold
	case CondStdBelow:
		return outS.Std < r.Threshold && outS.Std >= 0
	case CondAvgAbove:
		return math.Abs(outS.Avg) > r.Threshold
	case CondAvgBelow:
		return math.Abs(outS.Avg) < r.Threshold
	case CondMaxAbove:
		return math.Abs(outS.Max) > r.Threshold
	case CondActiveBelow:
		if outS.Total == 0 {
			return false
		}
		return float64(outS.Active)/float64(outS.Total) < r.Threshold
	case CondGainDrift:
		inMag := math.Abs(inS.Avg)
		if inMag < 1e-6 {
			return false
		}
		gain := math.Abs(outS.Avg) / inMag
		return math.Abs(gain-1) > r.Threshold
	default:
		return false
	}
}
