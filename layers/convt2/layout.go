package convt2

import (
	"fmt"

	"github.com/openfluke/welvet/core"
)

type layout struct {
	batch, inC, inH, inW, filters, outH, outW, kSize, stride, padding int
}

func parseLayout[T core.Numeric](cfg Config, input *core.Tensor[T]) (layout, error) {
	lay := layout{
		inC: cfg.InChannels, inH: cfg.Height, inW: cfg.Width, filters: cfg.Filters,
		outH: cfg.OutH(), outW: cfg.OutW(), kSize: cfg.Kernel, stride: cfg.Stride, padding: cfg.Padding,
	}
	if input == nil || len(input.Data) == 0 {
		return lay, fmt.Errorf("convt2: empty input")
	}
	if len(input.Shape) != 4 {
		return lay, fmt.Errorf("convt2: shape need [batch,inChannels,H,W], got %v", input.Shape)
	}
	lay.batch = input.Shape[0]
	if input.Shape[1] != lay.inC {
		return lay, fmt.Errorf("convt2: channels %d != %d", input.Shape[1], lay.inC)
	}
	if input.Shape[2] != lay.inH || input.Shape[3] != lay.inW {
		return lay, fmt.Errorf("convt2: spatial %dx%d != %dx%d", input.Shape[2], input.Shape[3], lay.inH, lay.inW)
	}
	if lay.batch <= 0 {
		return lay, fmt.Errorf("convt2: invalid batch")
	}
	want := lay.batch * lay.inC * lay.inH * lay.inW
	if len(input.Data) < want {
		return lay, fmt.Errorf("convt2: data len %d < %d", len(input.Data), want)
	}
	return lay, nil
}

func spatialOut(lay layout) int { return lay.outH * lay.outW }
func spatialIn(lay layout) int  { return lay.inH * lay.inW }
