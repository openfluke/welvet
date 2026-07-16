package embedding

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/dense"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/weights"
)

// Layer is a vocab×emb embedding table.
type Layer struct {
	Core    core.Layer
	Cfg     Config
	Exec    core.ExecConfig
	Weights *weights.Store // shape VocabSize × EmbeddingDim
}

// New creates Embedding with small deterministic init, FormatNone Float32.
func New(cfg Config) (*Layer, error) {
	init := make([]float32, cfg.WeightCount())
	for i := range init {
		init[i] = float32((i%5)-2) * 0.02
	}
	return NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone, init)
}

// NewConfigured builds Embedding from init (length Vocab×Emb).
func NewConfigured[T core.Numeric](cfg Config, dt core.DType, format quant.Format, init []T) (*Layer, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	ws, err := weights.New(cfg.VocabSize, cfg.EmbeddingDim, init, dt, format)
	if err != nil {
		return nil, fmt.Errorf("embedding weights: %w", err)
	}
	return &Layer{
		Core: core.Layer{
			Type:         core.LayerEmbedding,
			DType:        dt,
			Activation:   core.ActivationLinear,
			InputHeight:  cfg.VocabSize,
			OutputHeight: cfg.EmbeddingDim,
			TileSize:     32,
			MultiCore:    true,
		},
		Cfg:     cfg,
		Weights: ws,
		Exec: core.ExecConfig{
			Backend:   core.BackendCPUTiled,
			MultiCore: true,
			TileSize:  32,
		},
	}, nil
}

// SetDType sets table dtype.
func (l *Layer) SetDType(dt core.DType) error {
	if l == nil || l.Weights == nil {
		return fmt.Errorf("embedding: nil")
	}
	if err := l.Weights.SetDType(dt); err != nil {
		return err
	}
	l.Core.DType = dt
	return nil
}

// Pack packs the table to format.
func (l *Layer) Pack(format quant.Format) error {
	if l == nil || l.Weights == nil {
		return fmt.Errorf("embedding: nil")
	}
	return l.Weights.Pack(format)
}

// Forward dispatches by Exec.Backend.
func Forward[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if l == nil || l.Weights == nil || input == nil {
		return nil, nil, fmt.Errorf("embedding: nil layer/input")
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

// Backward dispatches; gradW is [vocab, emb]; gradIn is zeros (same shape as input).
func Backward[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if l == nil || l.Weights == nil {
		return nil, nil, fmt.Errorf("embedding: nil layer")
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

// GradWSize is vocab × emb.
func (l *Layer) GradWSize() int {
	if l == nil || l.Weights == nil {
		return 0
	}
	return l.Weights.Rows * l.Weights.Cols
}

// PermutationOK — same coverage matrix as Dense weight cells.
func PermutationOK(dt core.DType, format quant.Format, backend core.Backend) bool {
	return dense.PermutationOK(dt, format, backend)
}

// AllPermutations lists the Embedding coverage matrix.
func AllPermutations() (out []struct {
	DType   core.DType
	Format  quant.Format
	Backend core.Backend
}) {
	return dense.AllPermutations()
}
