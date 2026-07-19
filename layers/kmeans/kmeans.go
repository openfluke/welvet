package kmeans

import (
	"fmt"
	"math"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
)

// Layer is soft K-Means. Centers live on Dense (Rows=K, Cols=FeatureDim).
type Layer struct {
	Core    core.Layer
	Cfg     Config
	Exec    core.ExecConfig
	Centers *dense.Layer
}

// New creates KMeans with FormatNone Float32 zeros.
func New(cfg Config) (*Layer, error) {
	return NewConfigured[float32](cfg, core.DTypeFloat32, quant.FormatNone, nil)
}

// NewConfigured builds centers from flat [K×D] init.
func NewConfigured[T core.Numeric](cfg Config, dt core.DType, format quant.Format, init []T) (*Layer, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	ctr, err := dense.NewConfigured(cfg.FeatureDim, cfg.NumClusters, core.ActivationLinear, dt, format, init)
	if err != nil {
		return nil, fmt.Errorf("kmeans centers: %w", err)
	}
	out := cfg.OutDim()
	l := &Layer{
		Core: core.Layer{
			Type:         core.LayerKMeans,
			DType:        dt,
			Activation:   cfg.Activation,
			InputHeight:  cfg.FeatureDim,
			OutputHeight: out,
			TileSize:     32,
			MultiCore:    true,
		},
		Cfg:     cfg,
		Centers: ctr,
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
	if l == nil || l.Centers == nil {
		return
	}
	l.Centers.Exec = l.Exec
	l.Centers.Core.TileSize = l.Core.TileSize
	l.Centers.Core.MultiCore = l.Core.MultiCore
	l.Core.Activation = l.Cfg.Activation
}

func (l *Layer) SetDType(dt core.DType) error {
	if l == nil || l.Centers == nil {
		return fmt.Errorf("kmeans: nil")
	}
	if err := l.Centers.Weights.SetDType(dt); err != nil {
		return err
	}
	l.Core.DType = dt
	l.Centers.Core.DType = dt
	return nil
}

func (l *Layer) Pack(format quant.Format) error {
	if l == nil || l.Centers == nil {
		return fmt.Errorf("kmeans: nil")
	}
	return l.Centers.Weights.Pack(format)
}

func Forward[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if l == nil || l.Centers == nil || input == nil {
		return nil, nil, fmt.Errorf("kmeans: nil layer/input")
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
	if l == nil || l.Centers == nil {
		return nil, nil, fmt.Errorf("kmeans: nil layer")
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
	if l == nil || l.Centers == nil || l.Centers.Weights == nil {
		return 0
	}
	return l.Centers.Weights.Rows * l.Centers.Weights.Cols
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

func softMax(logits []float64) []float64 {
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
	for i := range out {
		out[i] /= sum
	}
	return out
}

func softMaxBwd(gy, y []float64) []float64 {
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
