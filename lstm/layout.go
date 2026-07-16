package lstm

import (
	"fmt"

	"github.com/openfluke/welvet/core"
)

type layout struct {
	batch, seq, in, hid int
}

func parseLayout[T core.Numeric](cfg Config, input *core.Tensor[T]) (layout, error) {
	lay := layout{in: cfg.InputSize, hid: cfg.HiddenSize, seq: cfg.SeqLen}
	if input == nil || len(input.Data) == 0 {
		return lay, fmt.Errorf("lstm: empty input")
	}
	if len(input.Shape) != 3 {
		return lay, fmt.Errorf("lstm: shape need [batch,seq,input], got %v", input.Shape)
	}
	lay.batch = input.Shape[0]
	if input.Shape[1] != lay.seq {
		return lay, fmt.Errorf("lstm: seq %d != %d", input.Shape[1], lay.seq)
	}
	if input.Shape[2] != lay.in {
		return lay, fmt.Errorf("lstm: input dim %d != %d", input.Shape[2], lay.in)
	}
	if lay.batch <= 0 {
		return lay, fmt.Errorf("lstm: invalid batch")
	}
	want := lay.batch * lay.seq * lay.in
	if len(input.Data) < want {
		return lay, fmt.Errorf("lstm: data len %d < %d", len(input.Data), want)
	}
	return lay, nil
}

func xAt[T core.Numeric](input *core.Tensor[T], lay layout, t int) *core.Tensor[T] {
	out := core.NewTensor[T](lay.batch, lay.in)
	for b := 0; b < lay.batch; b++ {
		copy(out.Data[b*lay.in:(b+1)*lay.in], input.Data[b*lay.seq*lay.in+t*lay.in:b*lay.seq*lay.in+(t+1)*lay.in])
	}
	return out
}

func hTensor[T core.Numeric](h []T, batch, hid int) *core.Tensor[T] {
	out := core.NewTensor[T](batch, hid)
	copy(out.Data, h)
	return out
}
