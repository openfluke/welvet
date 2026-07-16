package swiglu

import (
	"fmt"

	"github.com/openfluke/welvet/core"
)

// layout describes [batch, in] or [batch, seq, in] → flat [batch*seq, in].
type layout struct {
	batch, seq, in int
	elemStride     int
}

func parseLayout[T core.Numeric](inDim int, input *core.Tensor[T]) (layout, error) {
	lay := layout{in: inDim}
	if input == nil || len(input.Data) == 0 {
		return lay, fmt.Errorf("swiglu: empty input")
	}
	switch len(input.Shape) {
	case 2:
		lay.batch = input.Shape[0]
		lay.seq = 1
		if input.Shape[1] != inDim {
			return lay, fmt.Errorf("swiglu: width %d != InputDim %d", input.Shape[1], inDim)
		}
	case 3:
		lay.batch = input.Shape[0]
		lay.seq = input.Shape[1]
		if input.Shape[2] != inDim {
			return lay, fmt.Errorf("swiglu: last dim %d != InputDim %d", input.Shape[2], inDim)
		}
	default:
		return lay, fmt.Errorf("swiglu: shape need [batch,in] or [batch,seq,in], got %v", input.Shape)
	}
	if lay.batch <= 0 || lay.seq <= 0 {
		return lay, fmt.Errorf("swiglu: invalid batch/seq")
	}
	lay.elemStride = lay.seq * lay.in
	want := lay.batch * lay.elemStride
	if len(input.Data) < want {
		return lay, fmt.Errorf("swiglu: data len %d < %d", len(input.Data), want)
	}
	return lay, nil
}

func flatten[T core.Numeric](input *core.Tensor[T], lay layout) *core.Tensor[T] {
	bs := lay.batch * lay.seq
	out := core.NewTensor[T](bs, lay.in)
	copy(out.Data, input.Data[:bs*lay.in])
	return out
}

func unflatten[T core.Numeric](flat *core.Tensor[T], lay layout, last int) *core.Tensor[T] {
	if lay.seq == 1 {
		out := core.NewTensor[T](lay.batch, last)
		copy(out.Data, flat.Data[:lay.batch*last])
		return out
	}
	out := core.NewTensor[T](lay.batch, lay.seq, last)
	copy(out.Data, flat.Data[:lay.batch*lay.seq*last])
	return out
}
