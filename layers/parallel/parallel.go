package parallel

import (
	"fmt"
	"math"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
)

// Layer is Parallel / MoE over Dense branches.
type Layer struct {
	Core     core.Layer
	Cfg      Config
	Exec     core.ExecConfig
	Branches []*dense.Layer
	Gate     *dense.Layer // Rows=Branches, Cols=Dim; used when CombineFilter
}

// New creates Parallel with FormatNone Float32 zeros.
func New(cfg Config) (*Layer, error) {
	return NewConfigured[float32](cfg, core.DTypeFloat32, quant.FormatNone, nil, nil)
}

// NewConfigured builds branches. packed is optional concat of branch [OutFeat×Dim] weights;
// gateInit is optional [Branches×Dim] for filter mode.
func NewConfigured[T core.Numeric](cfg Config, dt core.DType, format quant.Format, packed, gateInit []T) (*Layer, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	childN := cfg.OutFeat * cfg.Dim
	branches := make([]*dense.Layer, cfg.Branches)
	for i := 0; i < cfg.Branches; i++ {
		var init []T
		if packed != nil {
			off := i * childN
			if len(packed) < off+childN {
				return nil, fmt.Errorf("parallel: packed short at branch %d", i)
			}
			init = packed[off : off+childN]
		}
		ch, err := dense.NewConfigured(cfg.Dim, cfg.OutFeat, core.ActivationLinear, dt, format, init)
		if err != nil {
			return nil, fmt.Errorf("parallel branch %d: %w", i, err)
		}
		branches[i] = ch
	}
	var gate *dense.Layer
	if cfg.Combine == CombineFilter {
		g, err := dense.NewConfigured(cfg.Dim, cfg.Branches, core.ActivationLinear, dt, format, gateInit)
		if err != nil {
			return nil, fmt.Errorf("parallel gate: %w", err)
		}
		gate = g
	}
	out := cfg.OutDim()
	l := &Layer{
		Core: core.Layer{
			Type:         core.LayerParallel,
			DType:        dt,
			Activation:   core.ActivationLinear,
			InputHeight:  cfg.Dim,
			OutputHeight: out,
			TileSize:     32,
			MultiCore:    true,
		},
		Cfg:      cfg,
		Branches: branches,
		Gate:     gate,
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
	for _, ch := range l.Branches {
		if ch == nil {
			continue
		}
		ch.Exec = l.Exec
		ch.Core.TileSize = l.Core.TileSize
		ch.Core.MultiCore = l.Core.MultiCore
	}
	if l.Gate != nil {
		l.Gate.Exec = l.Exec
		l.Gate.Core.TileSize = l.Core.TileSize
		l.Gate.Core.MultiCore = l.Core.MultiCore
	}
}

// SetDType sets dtype on branches (+ gate).
func (l *Layer) SetDType(dt core.DType) error {
	if l == nil {
		return fmt.Errorf("parallel: nil")
	}
	for i, ch := range l.Branches {
		if err := ch.Weights.SetDType(dt); err != nil {
			return fmt.Errorf("parallel branch %d: %w", i, err)
		}
	}
	if l.Gate != nil {
		if err := l.Gate.Weights.SetDType(dt); err != nil {
			return fmt.Errorf("parallel gate: %w", err)
		}
	}
	l.Core.DType = dt
	return nil
}

// Pack packs branches (+ gate).
func (l *Layer) Pack(format quant.Format) error {
	if l == nil {
		return fmt.Errorf("parallel: nil")
	}
	for i, ch := range l.Branches {
		if err := ch.Weights.Pack(format); err != nil {
			return fmt.Errorf("parallel branch %d: %w", i, err)
		}
	}
	if l.Gate != nil {
		if err := l.Gate.Weights.Pack(format); err != nil {
			return fmt.Errorf("parallel gate: %w", err)
		}
	}
	return nil
}

// Forward dispatches by Exec.Backend.
func Forward[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if l == nil || input == nil {
		return nil, nil, fmt.Errorf("parallel: nil layer/input")
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

// Backward distributes grads; gradW is concat of branch dWs (+ gate dW for filter).
// pre.Nested holds branch preActs when available.
func Backward[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if l == nil {
		return nil, nil, fmt.Errorf("parallel: nil layer")
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

// GradWSize is sum of branch (+ gate) weight lengths.
func (l *Layer) GradWSize() int {
	if l == nil {
		return 0
	}
	n := 0
	for _, ch := range l.Branches {
		if ch != nil && ch.Weights != nil {
			n += ch.Weights.Rows * ch.Weights.Cols
		}
	}
	if l.Gate != nil && l.Gate.Weights != nil {
		n += l.Gate.Weights.Rows * l.Gate.Weights.Cols
	}
	return n
}

// PermutationOK — same as Dense children.
func PermutationOK(dt core.DType, format quant.Format, backend core.Backend) bool {
	return dense.PermutationOK(dt, format, backend)
}

// AllPermutations lists the coverage matrix.
func AllPermutations() (out []struct {
	DType   core.DType
	Format  quant.Format
	Backend core.Backend
}) {
	return dense.AllPermutations()
}

func softmaxF64(logits []float64) []float64 {
	maxZ := logits[0]
	for _, z := range logits[1:] {
		if z > maxZ {
			maxZ = z
		}
	}
	out := make([]float64, len(logits))
	var sum float64
	for i, z := range logits {
		e := math.Exp(z - maxZ)
		out[i] = e
		sum += e
	}
	if sum == 0 {
		inv := 1.0 / float64(len(out))
		for i := range out {
			out[i] = inv
		}
		return out
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

func softmaxBwd(gy, y []float64) []float64 {
	var dot float64
	for i := range y {
		dot += gy[i] * y[i]
	}
	out := make([]float64, len(y))
	for i := range y {
		out[i] = y[i] * (gy[i] - dot)
	}
	return out
}
