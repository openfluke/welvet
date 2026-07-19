package convt1

import (
	"fmt"

	"github.com/openfluke/welvet/core"
)

type layout struct {
	batch, inC, seq, filters, outLen, kSize, stride, padding int
}

func parseLayout[T core.Numeric](cfg Config, input *core.Tensor[T]) (layout, error) {
	lay := layout{
		inC: cfg.InChannels, seq: cfg.SeqLen, filters: cfg.Filters,
		outLen: cfg.OutLen(), kSize: cfg.Kernel, stride: cfg.Stride, padding: cfg.Padding,
	}
	if input == nil || len(input.Data) == 0 {
		return lay, fmt.Errorf("convt1: empty input")
	}
	if len(input.Shape) != 3 {
		return lay, fmt.Errorf("convt1: shape need [batch,inChannels,seqLen], got %v", input.Shape)
	}
	lay.batch = input.Shape[0]
	if input.Shape[1] != lay.inC {
		return lay, fmt.Errorf("convt1: channels %d != %d", input.Shape[1], lay.inC)
	}
	if input.Shape[2] != lay.seq {
		return lay, fmt.Errorf("convt1: seq %d != %d", input.Shape[2], lay.seq)
	}
	if lay.batch <= 0 {
		return lay, fmt.Errorf("convt1: invalid batch")
	}
	want := lay.batch * lay.inC * lay.seq
	if len(input.Data) < want {
		return lay, fmt.Errorf("convt1: data len %d < %d", len(input.Data), want)
	}
	return lay, nil
}
