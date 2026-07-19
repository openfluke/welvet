package convt3

import (
	"fmt"

	"github.com/openfluke/welvet/core"
)

type layout struct {
	batch, inC, inD, inH, inW, filters, outD, outH, outW, kSize, stride, padding int
}

func parseLayout[T core.Numeric](cfg Config, input *core.Tensor[T]) (layout, error) {
	lay := layout{
		inC: cfg.InChannels, inD: cfg.Depth, inH: cfg.Height, inW: cfg.Width, filters: cfg.Filters,
		outD: cfg.OutD(), outH: cfg.OutH(), outW: cfg.OutW(), kSize: cfg.Kernel, stride: cfg.Stride, padding: cfg.Padding,
	}
	if input == nil || len(input.Data) == 0 {
		return lay, fmt.Errorf("convt3: empty input")
	}
	if len(input.Shape) != 5 {
		return lay, fmt.Errorf("convt3: shape need [batch,inChannels,D,H,W], got %v", input.Shape)
	}
	lay.batch = input.Shape[0]
	if input.Shape[1] != lay.inC {
		return lay, fmt.Errorf("convt3: channels %d != %d", input.Shape[1], lay.inC)
	}
	if input.Shape[2] != lay.inD || input.Shape[3] != lay.inH || input.Shape[4] != lay.inW {
		return lay, fmt.Errorf("convt3: spatial mismatch")
	}
	if lay.batch <= 0 {
		return lay, fmt.Errorf("convt3: invalid batch")
	}
	want := lay.batch * lay.inC * lay.inD * lay.inH * lay.inW
	if len(input.Data) < want {
		return lay, fmt.Errorf("convt3: data len %d < %d", len(input.Data), want)
	}
	return lay, nil
}

func spatialOut(lay layout) int { return lay.outD * lay.outH * lay.outW }
func spatialIn(lay layout) int  { return lay.inD * lay.inH * lay.inW }
