// Package backward runs the reverse volumetric pass using a forward.Result tape.
package backward

import (
	"fmt"
	"time"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/cnn2"
	"github.com/openfluke/welvet/layers/cnn3"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/embedding"
	"github.com/openfluke/welvet/forward"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/lstm"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/rnn"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/layers/softmax"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/tanhi"
)

// GradW is one cell's weight gradient (Dense / MHA concat, …).
type GradW[T core.Numeric] struct {
	Coord architecture.Coord
	DW    *core.Tensor[T]
}

// Result holds gradIn at the network input plus per-cell dW.
type Result[T core.Numeric] struct {
	GradIn *core.Tensor[T]
	GradWs []GradW[T]
}

// Backward runs steps in reverse order. Remote-link cells receive grad into their
// local tape input only (grad is not re-routed to the remote source in v0).
func Backward[T core.Numeric](fwd *forward.Result[T], gradOut *core.Tensor[T]) (*Result[T], error) {
	if fwd == nil || len(fwd.Steps) == 0 || gradOut == nil {
		return nil, fmt.Errorf("backward: nil/empty tape or gradOut")
	}
	gy := gradOut
	out := &Result[T]{GradWs: make([]GradW[T], 0, len(fwd.Steps))}
	tanhiCfg := tanhi.ConfigFromGrid(fwd.Grid)

	for i := len(fwd.Steps) - 1; i >= 0; i-- {
		st := fwd.Steps[i]
		t0 := time.Now()
		gx, dw, err := dispatchBwd[T](st, gy)
		t1 := time.Now()
		if err != nil {
			return nil, fmt.Errorf("backward %v: %w", st.Coord, err)
		}
		if dw != nil {
			out.GradWs = append(out.GradWs, GradW[T]{Coord: st.Coord, DW: dw})
		}
		var shape []int
		if tanhiCfg != nil && tanhiCfg.SendShape && gx != nil {
			shape = append([]int(nil), gx.Shape...)
		}
		tanhi.Emit(tanhiCfg, "bwd", i, st.Cell, t0, t1, shape)
		gy = gx
	}
	out.GradIn = gy
	return out, nil
}

// Cell dispatches one cell Op backward (used by volumetric step mesh).
func Cell[T core.Numeric](cell *architecture.Cell, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if cell == nil {
		return nil, nil, fmt.Errorf("backward: nil cell")
	}
	return dispatchBwd(forward.Step[T]{Cell: cell, Input: input, Pre: pre}, gradOut)
}

func dispatchBwd[T core.Numeric](st forward.Step[T], gradOut *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if st.Cell == nil {
		return nil, nil, fmt.Errorf("nil cell")
	}
	switch st.Cell.Layer.Type {
	case core.LayerDense:
		dl, ok := st.Cell.Op.(*dense.Layer)
		if !ok || dl == nil {
			return nil, nil, fmt.Errorf("dense cell Op is %T", st.Cell.Op)
		}
		return dense.Backward(dl, gradOut, st.Input, st.Pre)
	case core.LayerMultiHeadAttention:
		ml, ok := st.Cell.Op.(*mha.Layer)
		if !ok || ml == nil {
			return nil, nil, fmt.Errorf("mha cell Op is %T", st.Cell.Op)
		}
		return mha.Backward(ml, gradOut, st.Input, st.Pre)
	case core.LayerSwiGLU:
		sl, ok := st.Cell.Op.(*swiglu.Layer)
		if !ok || sl == nil {
			return nil, nil, fmt.Errorf("swiglu cell Op is %T", st.Cell.Op)
		}
		return swiglu.Backward(sl, gradOut, st.Input, st.Pre)
	case core.LayerRMSNorm:
		rl, ok := st.Cell.Op.(*rmsnorm.Layer)
		if !ok || rl == nil {
			return nil, nil, fmt.Errorf("rmsnorm cell Op is %T", st.Cell.Op)
		}
		return rmsnorm.Backward(rl, gradOut, st.Input, st.Pre)
	case core.LayerLayerNorm:
		ll, ok := st.Cell.Op.(*layernorm.Layer)
		if !ok || ll == nil {
			return nil, nil, fmt.Errorf("layernorm cell Op is %T", st.Cell.Op)
		}
		return layernorm.Backward(ll, gradOut, st.Input, st.Pre)
	case core.LayerCNN1:
		cl, ok := st.Cell.Op.(*cnn1.Layer)
		if !ok || cl == nil {
			return nil, nil, fmt.Errorf("cnn1 cell Op is %T", st.Cell.Op)
		}
		return cnn1.Backward(cl, gradOut, st.Input, st.Pre)
	case core.LayerCNN2:
		cl, ok := st.Cell.Op.(*cnn2.Layer)
		if !ok || cl == nil {
			return nil, nil, fmt.Errorf("cnn2 cell Op is %T", st.Cell.Op)
		}
		return cnn2.Backward(cl, gradOut, st.Input, st.Pre)
	case core.LayerCNN3:
		cl, ok := st.Cell.Op.(*cnn3.Layer)
		if !ok || cl == nil {
			return nil, nil, fmt.Errorf("cnn3 cell Op is %T", st.Cell.Op)
		}
		return cnn3.Backward(cl, gradOut, st.Input, st.Pre)
	case core.LayerRNN:
		rl, ok := st.Cell.Op.(*rnn.Layer)
		if !ok || rl == nil {
			return nil, nil, fmt.Errorf("rnn cell Op is %T", st.Cell.Op)
		}
		return rnn.Backward(rl, gradOut, st.Input, st.Pre)
	case core.LayerLSTM:
		ll, ok := st.Cell.Op.(*lstm.Layer)
		if !ok || ll == nil {
			return nil, nil, fmt.Errorf("lstm cell Op is %T", st.Cell.Op)
		}
		return lstm.Backward(ll, gradOut, st.Input, st.Pre)
	case core.LayerEmbedding:
		el, ok := st.Cell.Op.(*embedding.Layer)
		if !ok || el == nil {
			return nil, nil, fmt.Errorf("embedding cell Op is %T", st.Cell.Op)
		}
		return embedding.Backward(el, gradOut, st.Input, st.Pre)
	case core.LayerSoftmax:
		sl, ok := st.Cell.Op.(*softmax.Layer)
		if !ok || sl == nil {
			return nil, nil, fmt.Errorf("softmax cell Op is %T", st.Cell.Op)
		}
		return softmax.Backward(sl, gradOut, st.Input, st.Pre)
	case core.LayerSequential:
		ql, ok := st.Cell.Op.(*sequential.Layer)
		if !ok || ql == nil {
			return nil, nil, fmt.Errorf("sequential cell Op is %T", st.Cell.Op)
		}
		return sequential.Backward(ql, gradOut, st.Input, st.Pre)
	case core.LayerResidual:
		rl, ok := st.Cell.Op.(*residual.Layer)
		if !ok || rl == nil {
			return nil, nil, fmt.Errorf("residual cell Op is %T", st.Cell.Op)
		}
		return residual.Backward(rl, gradOut, st.Input, st.Pre)
	default:
		return nil, nil, fmt.Errorf("unsupported layer type %s", st.Cell.Layer.Type)
	}
}
