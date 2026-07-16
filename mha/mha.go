package mha

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/dense"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/seqmix"
)

// Layer is multi-head attention. Projections Q/K/V/O are dense.Layer units
// (dtype / quant / backend coverage rides Dense).
//
// Policy (Mask / Pos / Mode) makes the same Layer cover decoder LM, encoder,
// diffusion self/cross, Prefix-LM, and sliding-window variants.
type Layer struct {
	Core core.Layer
	Cfg  Config
	Exec core.ExecConfig

	Q, K, V, O *dense.Layer

	// Context holds cross-attn K/V source [batch, ctx_seq, d_model] (ModeCross).
	// Set via SetContext or ForwardWithContext.
	Context any // *core.Tensor[T] — typed at call time

	// CustomAllow is required when Cfg.Mask == MaskCustom.
	// (qPos, kPos) absolute positions; return true to allow attention.
	CustomAllow func(qPos, kPos int) bool

	// Optional Q/K RMSNorm scales (Qwen-style); length HeadDim or NumHeads*HeadDim.
	QNormWeight []float64
	KNormWeight []float64

	// KV cache (host)
	KVCacheK []float64
	KVCacheV []float64
	KVOffset int
}

// MixerKind identifies this layer under the seqmix contract.
func (l *Layer) MixerKind() seqmix.Kind { return seqmix.KindAttention }

// New creates MHA with zero FormatNone weights.
func New(cfg Config) (*Layer, error) {
	return NewConfigured[float64](cfg, core.DTypeFloat32, quant.FormatNone, nil, nil, nil, nil)
}

// NewConfigured builds MHA with optional init weights for Q/K/V/O (row-major [out×in]).
func NewConfigured[T core.Numeric](
	cfg Config,
	dt core.DType,
	format quant.Format,
	qInit, kInit, vInit, oInit []T,
) (*Layer, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	qDim, kvDim, d := cfg.QDim(), cfg.KVDim(), cfg.DModel

	Q, err := dense.NewConfigured(d, qDim, core.ActivationLinear, dt, format, qInit)
	if err != nil {
		return nil, fmt.Errorf("mha Q: %w", err)
	}
	K, err := dense.NewConfigured(d, kvDim, core.ActivationLinear, dt, format, kInit)
	if err != nil {
		return nil, fmt.Errorf("mha K: %w", err)
	}
	V, err := dense.NewConfigured(d, kvDim, core.ActivationLinear, dt, format, vInit)
	if err != nil {
		return nil, fmt.Errorf("mha V: %w", err)
	}
	O, err := dense.NewConfigured(qDim, d, core.ActivationLinear, dt, format, oInit)
	if err != nil {
		return nil, fmt.Errorf("mha O: %w", err)
	}

	l := &Layer{
		Core: core.Layer{
			Type:         core.LayerMultiHeadAttention,
			DType:        dt,
			Activation:   core.ActivationLinear,
			InputHeight:  d,
			OutputHeight: d,
			TileSize:     32,
			MultiCore:    true,
		},
		Cfg: cfg,
		Exec: core.ExecConfig{
			Backend:   core.BackendCPUTiled,
			MultiCore: true,
			TileSize:  32,
		},
		Q: Q, K: K, V: V, O: O,
	}
	l.syncProjExec()
	return l, nil
}

func (l *Layer) syncProjExec() {
	if l == nil {
		return
	}
	for _, p := range []*dense.Layer{l.Q, l.K, l.V, l.O} {
		if p != nil {
			p.Exec = l.Exec
			p.Core.TileSize = l.Core.TileSize
			p.Core.MultiCore = l.Core.MultiCore
		}
	}
}

// SetContext stores cross-attn conditioning (encoder / diffusion cond tokens).
func SetContext[T core.Numeric](l *Layer, ctx *core.Tensor[T]) error {
	if l == nil {
		return fmt.Errorf("mha: nil layer")
	}
	if l.Cfg.Mode != ModeCross {
		return fmt.Errorf("mha: SetContext requires ModeCross (got %s)", l.Cfg.Mode)
	}
	if ctx == nil {
		return fmt.Errorf("mha: nil context")
	}
	l.Context = ctx
	return nil
}

// ClearContext drops stored cross-attn context.
func (l *Layer) ClearContext() {
	if l != nil {
		l.Context = nil
	}
}

// SetDType sets weight dtype on all four projections.
func (l *Layer) SetDType(dt core.DType) error {
	if l == nil {
		return fmt.Errorf("mha: nil")
	}
	for _, p := range []*dense.Layer{l.Q, l.K, l.V, l.O} {
		if err := p.Weights.SetDType(dt); err != nil {
			return err
		}
	}
	l.Core.DType = dt
	return nil
}

// Pack packs all four projections to format.
func (l *Layer) Pack(format quant.Format) error {
	if l == nil {
		return fmt.Errorf("mha: nil")
	}
	for _, p := range []*dense.Layer{l.Q, l.K, l.V, l.O} {
		if err := p.Weights.Pack(format); err != nil {
			return err
		}
	}
	return nil
}

// Forward dispatches by Exec.Backend (self-attn, or cross if Context already set).
func Forward[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if l == nil || l.Q == nil || input == nil {
		return nil, nil, fmt.Errorf("mha: nil layer/input")
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

// ForwardWithContext runs cross-attn with an explicit context tensor.
func ForwardWithContext[T core.Numeric](l *Layer, input, context *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if err := SetContext(l, context); err != nil {
		return nil, nil, err
	}
	return Forward(l, input)
}

// Backward dispatches by Exec.Backend.
func Backward[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if l == nil || l.Q == nil {
		return nil, nil, fmt.Errorf("mha: nil layer")
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

// GradWSize is the concatenated Q+K+V+O weight gradient length (no biases).
func (l *Layer) GradWSize() int {
	if l == nil {
		return 0
	}
	n := 0
	for _, p := range []*dense.Layer{l.Q, l.K, l.V, l.O} {
		if p != nil && p.Weights != nil {
			n += p.Weights.Rows * p.Weights.Cols
		}
	}
	return n
}

func (l *Layer) allowFn() (func(q, k int) bool, error) {
	if l.Cfg.Mask == MaskCustom {
		if l.CustomAllow == nil {
			return nil, fmt.Errorf("mha: MaskCustom requires Layer.CustomAllow")
		}
		return l.CustomAllow, nil
	}
	cfg := l.Cfg
	return func(q, k int) bool { return Allow(cfg, q, k) }, nil
}

// PermutationOK — same coverage as Dense projections.
func PermutationOK(dt core.DType, format quant.Format, backend core.Backend) bool {
	return dense.PermutationOK(dt, format, backend)
}

// AllPermutations lists the MHA coverage matrix (projection cells).
func AllPermutations() (out []struct {
	DType   core.DType
	Format  quant.Format
	Backend core.Backend
}) {
	return dense.AllPermutations()
}
