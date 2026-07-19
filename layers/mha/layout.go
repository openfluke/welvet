package mha

import (
	"fmt"

	"github.com/openfluke/welvet/core"
)

// layout describes [batch, seq, d_model] or [seq, d_model] indexing.
type layout struct {
	batch, seqLen, dModel int
	elemStride            int
}

func parseLayout[T core.Numeric](dModel int, input *core.Tensor[T]) (layout, error) {
	lay := layout{dModel: dModel}
	if input == nil || len(input.Data) == 0 {
		return lay, fmt.Errorf("mha: empty input")
	}
	switch len(input.Shape) {
	case 3:
		lay.batch = input.Shape[0]
		lay.seqLen = input.Shape[1]
		if input.Shape[2] != dModel {
			return lay, fmt.Errorf("mha: last dim %d != d_model %d", input.Shape[2], dModel)
		}
	case 2:
		lay.batch = 1
		if input.Shape[1] != dModel {
			return lay, fmt.Errorf("mha: width %d != d_model %d", input.Shape[1], dModel)
		}
		lay.seqLen = input.Shape[0]
	default:
		return lay, fmt.Errorf("mha: input shape need [batch,seq,d] or [seq,d], got %v", input.Shape)
	}
	if lay.batch <= 0 || lay.seqLen <= 0 {
		return lay, fmt.Errorf("mha: invalid batch/seq %d/%d", lay.batch, lay.seqLen)
	}
	lay.elemStride = lay.seqLen * lay.dModel
	want := lay.batch * lay.elemStride
	if len(input.Data) < want {
		return lay, fmt.Errorf("mha: data len %d < %d", len(input.Data), want)
	}
	return lay, nil
}

func (lay layout) base(b int) int { return b * lay.elemStride }

func (lay layout) idx(b, s, j int) int { return lay.base(b) + s*lay.dModel + j }

func flattenTokens[T core.Numeric](input *core.Tensor[T], lay layout) *core.Tensor[T] {
	bs := lay.batch * lay.seqLen
	// Already [tokens, d_model] contiguous — reuse (decode / flat forwards).
	if len(input.Shape) == 2 && input.Shape[0] == bs && input.Shape[1] == lay.dModel {
		return input
	}
	out := core.NewTensor[T](bs, lay.dModel)
	copy(out.Data, input.Data[:bs*lay.dModel])
	return out
}

func reshapeSeq[T core.Numeric](flat *core.Tensor[T], lay layout, last int) *core.Tensor[T] {
	n := lay.batch * lay.seqLen * last
	// Prefer view: same backing store, new shape (avoids copy on decode).
	if flat != nil && len(flat.Data) >= n {
		return &core.Tensor[T]{
			Shape: []int{lay.batch, lay.seqLen, last},
			Data:  flat.Data[:n],
		}
	}
	out := core.NewTensor[T](lay.batch, lay.seqLen, last)
	copy(out.Data, flat.Data[:n])
	return out
}
