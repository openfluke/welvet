// Package backward runs the reverse volumetric pass using a forward.Result tape.
package backward

import (
	"fmt"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/dense"
	"github.com/openfluke/welvet/forward"
	"github.com/openfluke/welvet/mha"
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

	for i := len(fwd.Steps) - 1; i >= 0; i-- {
		st := fwd.Steps[i]
		gx, dw, err := dispatchBwd[T](st, gy)
		if err != nil {
			return nil, fmt.Errorf("backward %v: %w", st.Coord, err)
		}
		if dw != nil {
			out.GradWs = append(out.GradWs, GradW[T]{Coord: st.Coord, DW: dw})
		}
		gy = gx
	}
	out.GradIn = gy
	return out, nil
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
	default:
		return nil, nil, fmt.Errorf("unsupported layer type %s", st.Cell.Layer.Type)
	}
}
