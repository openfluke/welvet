package parallel

import (
	"fmt"
	"math"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
)

// Layer is Parallel / MoE over polymorphic branch Ops (any cell Op, including
// nested Parallel / Sequential). Gate remains Dense for filter mode.
type Layer struct {
	Core     core.Layer
	Cfg      Config
	Exec     core.ExecConfig
	Branches []any         // branch Ops (Dense, MHA, Parallel, …)
	Gate     *dense.Layer  // Rows=Branches, Cols=Dim; used when CombineFilter
}

// New creates Parallel with FormatNone Float32 zeros (Dense branches).
func New(cfg Config) (*Layer, error) {
	return NewConfigured[float32](cfg, core.DTypeFloat32, quant.FormatNone, nil, nil)
}

// NewConfigured builds Dense branches. packed is optional concat of branch
// [OutFeat×Dim] weights; gateInit is optional [Branches×Dim] for filter mode.
func NewConfigured[T core.Numeric](cfg Config, dt core.DType, format quant.Format, packed, gateInit []T) (*Layer, error) {
	if cfg.OutFeat <= 0 {
		return nil, fmt.Errorf("parallel: NewConfigured requires OutFeat > 0")
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	childN := cfg.OutFeat * cfg.Dim
	branches := make([]any, cfg.Branches)
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
	return finishLayer(cfg, dt, branches, gate)
}

// NewFromBranches builds Parallel over arbitrary Ops. Each branch keeps its own
// dtype/format storage truth. cfg.Branches is set from len(branches) when 0.
// cfg.OutFeat may be 0 (widths measured at forward); when >0 it is kept for OutDim().
func NewFromBranches(cfg Config, branches []any, gate *dense.Layer) (*Layer, error) {
	if len(branches) == 0 {
		return nil, fmt.Errorf("parallel: NewFromBranches needs ≥1 branch")
	}
	for i, b := range branches {
		if b == nil {
			return nil, fmt.Errorf("parallel: nil branch at %d", i)
		}
	}
	if cfg.Branches == 0 {
		cfg.Branches = len(branches)
	}
	if cfg.Branches != len(branches) {
		return nil, fmt.Errorf("parallel: cfg.Branches=%d != len(branches)=%d", cfg.Branches, len(branches))
	}
	if cfg.Combine == "" {
		cfg.Combine = CombineConcat
	}
	if cfg.Combine == CombineFilter && gate == nil {
		return nil, fmt.Errorf("parallel: filter mode requires Gate")
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	dt := core.DTypeFloat32
	if gate != nil {
		dt = gate.Core.DType
	}
	return finishLayer(cfg, dt, append([]any(nil), branches...), gate)
}

func finishLayer(cfg Config, dt core.DType, branches []any, gate *dense.Layer) (*Layer, error) {
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
	l.SyncBranchExec()
	return l, nil
}

// SyncBranchExec copies Exec onto every branch Op (+ gate). Exported for dispatch.
func (l *Layer) SyncBranchExec() {
	if l == nil {
		return
	}
	for _, ch := range l.Branches {
		branchSyncExec(ch, l.Exec)
	}
	if l.Gate != nil {
		l.Gate.Exec = l.Exec
		l.Gate.Core.TileSize = l.Core.TileSize
		l.Gate.Core.MultiCore = l.Core.MultiCore
	}
}

func (l *Layer) syncChildExec() { l.SyncBranchExec() }

// SetDType sets dtype on every branch Op (+ gate).
func (l *Layer) SetDType(dt core.DType) error {
	if l == nil {
		return fmt.Errorf("parallel: nil")
	}
	for i, ch := range l.Branches {
		if err := branchSetDType(ch, dt); err != nil {
			return fmt.Errorf("parallel branch %d: %w", i, err)
		}
	}
	if l.Gate != nil {
		if err := l.Gate.SetDType(dt); err != nil {
			return fmt.Errorf("parallel gate: %w", err)
		}
	}
	l.Core.DType = dt
	return nil
}

// Pack packs every branch Op (+ gate).
func (l *Layer) Pack(format quant.Format) error {
	if l == nil {
		return fmt.Errorf("parallel: nil")
	}
	for i, ch := range l.Branches {
		if err := branchPack(ch, format); err != nil {
			return fmt.Errorf("parallel branch %d: %w", i, err)
		}
	}
	if l.Gate != nil {
		if err := l.Gate.Pack(format); err != nil {
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
	l.SyncBranchExec()
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
func Backward[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if l == nil {
		return nil, nil, fmt.Errorf("parallel: nil layer")
	}
	l.SyncBranchExec()
	switch l.Exec.Backend {
	case core.BackendSIMD:
		return BackwardSIMD(l, gradOut, input, pre)
	case core.BackendWebGPU:
		return BackwardWebGPU(l, gradOut, input, pre)
	default:
		return BackwardCPUTiled(l, gradOut, input, pre)
	}
}

// GradWSize is sum of branch (+ gate) GradWSize.
func (l *Layer) GradWSize() int {
	if l == nil {
		return 0
	}
	n := 0
	for _, ch := range l.Branches {
		n += branchGradWSize(ch)
	}
	if l.Gate != nil {
		n += l.Gate.GradWSize()
	}
	return n
}

// DenseBranch returns Branches[i] as *dense.Layer when the branch is Dense.
func (l *Layer) DenseBranch(i int) (*dense.Layer, bool) {
	if l == nil || i < 0 || i >= len(l.Branches) {
		return nil, false
	}
	d, ok := l.Branches[i].(*dense.Layer)
	return d, ok
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
