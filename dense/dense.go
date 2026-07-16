package dense

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/weights"
)

// Layer is a Dense unit. Activation tensors are never hardcoded — use Forward[T] / Backward[T].
type Layer struct {
	Core    core.Layer
	Weights *weights.Store
	Exec    core.ExecConfig

	// Reusable forward buffers (float32 LM path) — avoids NewTensor per GEMV.
	fwdOut []float32
}

// New creates a Dense layer with zero weights in the given dtype (FormatNone).
func New(in, out int, act core.ActivationType, dt core.DType) (*Layer, error) {
	ws, err := weights.New[float64](out, in, nil, dt, quant.FormatNone)
	if err != nil {
		return nil, err
	}
	return &Layer{
		Core: core.Layer{
			Type:         core.LayerDense,
			DType:        dt,
			Activation:   act,
			InputHeight:  in,
			OutputHeight: out,
			TileSize:     32,
			MultiCore:    true,
		},
		Weights: ws,
		Exec: core.ExecConfig{
			Backend:   core.BackendCPUTiled,
			MultiCore: true,
			TileSize:  32,
		},
	}, nil
}

// NewConfigured builds Dense from any Numeric init weights + dtype + quant format.
func NewConfigured[T core.Numeric](in, out int, act core.ActivationType, dt core.DType, format quant.Format, init []T) (*Layer, error) {
	ws, err := weights.New(out, in, init, dt, format)
	if err != nil {
		return nil, err
	}
	return &Layer{
		Core: core.Layer{
			Type:         core.LayerDense,
			DType:        dt,
			Activation:   act,
			InputHeight:  in,
			OutputHeight: out,
			TileSize:     32,
			MultiCore:    true,
		},
		Weights: ws,
		Exec: core.ExecConfig{
			Backend:   core.BackendCPUTiled,
			MultiCore: true,
			TileSize:  32,
		},
	}, nil
}

// Forward dispatches by Exec.Backend for activation dtype T (not hardcoded).
func Forward[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if l == nil || l.Weights == nil || input == nil {
		return nil, nil, fmt.Errorf("dense: nil layer/input")
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

// Backward dispatches by Exec.Backend for activation dtype T.
func Backward[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if l == nil || l.Weights == nil {
		return nil, nil, fmt.Errorf("dense: nil layer")
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

func dims[T core.Numeric](l *Layer, input *core.Tensor[T]) (batch, in, out int, err error) {
	in = l.Core.InputHeight
	out = l.Core.OutputHeight
	if len(input.Shape) < 2 {
		return 0, 0, 0, fmt.Errorf("dense: input shape need [batch,in]")
	}
	batch = input.Shape[0]
	if input.Shape[1] != in {
		return 0, 0, 0, fmt.Errorf("dense: input width %d != layer in %d", input.Shape[1], in)
	}
	if l.Weights.Rows != out || l.Weights.Cols != in {
		return 0, 0, 0, fmt.Errorf("dense: weight shape %dx%d != %dx%d", l.Weights.Rows, l.Weights.Cols, out, in)
	}
	return batch, in, out, nil
}

func applyBiasAct[T core.Numeric](pre, post []T, bias []float64, act core.ActivationType, batch, out int) {
	n := batch * out
	if n == 0 {
		return
	}
	if act == core.ActivationLinear && (bias == nil || len(bias) == 0) {
		if len(post) >= n && (len(pre) < 1 || len(post) < 1 || &pre[0] != &post[0]) {
			copy(post[:n], pre[:n])
		}
		return
	}
	for b := 0; b < batch; b++ {
		for o := 0; o < out; o++ {
			i := b*out + o
			v := core.AsFloat64(pre[i])
			if bias != nil && o < len(bias) {
				v += bias[o]
				pre[i] = core.FromFloat64[T](v)
			}
			post[i] = core.Activate(core.FromFloat64[T](v), act)
		}
	}
}

// linearNoBias reports whether forward can alias post=pre and skip applyBiasAct work.
func linearNoBias(l *Layer) bool {
	return l != nil && l.Core.Activation == core.ActivationLinear &&
		(l.Weights == nil || len(l.Weights.Bias) == 0)
}
