package layernorm

import (
	"fmt"

	"github.com/openfluke/welvet/core"
)

type layout struct {
	batch, seq, dim int
}

func parseLayout[T core.Numeric](dim int, input *core.Tensor[T]) (layout, error) {
	lay := layout{dim: dim}
	if input == nil || len(input.Data) == 0 {
		return lay, fmt.Errorf("layernorm: empty input")
	}
	switch len(input.Shape) {
	case 2:
		lay.batch = input.Shape[0]
		lay.seq = 1
		if input.Shape[1] != dim {
			return lay, fmt.Errorf("layernorm: width %d != Dim %d", input.Shape[1], dim)
		}
	case 3:
		lay.batch = input.Shape[0]
		lay.seq = input.Shape[1]
		if input.Shape[2] != dim {
			return lay, fmt.Errorf("layernorm: last dim %d != Dim %d", input.Shape[2], dim)
		}
	default:
		return lay, fmt.Errorf("layernorm: shape need [batch,dim] or [batch,seq,dim], got %v", input.Shape)
	}
	if lay.batch <= 0 || lay.seq <= 0 {
		return lay, fmt.Errorf("layernorm: invalid batch/seq")
	}
	want := lay.batch * lay.seq * dim
	if len(input.Data) < want {
		return lay, fmt.Errorf("layernorm: data len %d < %d", len(input.Data), want)
	}
	return lay, nil
}

func tokens(lay layout) int { return lay.batch * lay.seq }

func reshapeOut[T core.Numeric](flat []T, lay layout) *core.Tensor[T] {
	if lay.seq == 1 {
		out := core.NewTensor[T](lay.batch, lay.dim)
		copy(out.Data, flat)
		return out
	}
	out := core.NewTensor[T](lay.batch, lay.seq, lay.dim)
	copy(out.Data, flat)
	return out
}
