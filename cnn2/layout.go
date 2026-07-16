package cnn2

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
		return lay, fmt.Errorf("cnn2: empty input")
	}
	if len(input.Shape) != 4 {
		return lay, fmt.Errorf("cnn2: shape need [batch,inChannels,H,W], got %v", input.Shape)
	}
	lay.batch = input.Shape[0]
	if input.Shape[1] != lay.inC {
		return lay, fmt.Errorf("cnn2: channels %d != %d", input.Shape[1], lay.inC)
	}
	if input.Shape[2] != lay.inH || input.Shape[3] != lay.inW {
		return lay, fmt.Errorf("cnn2: spatial %dx%d != %dx%d", input.Shape[2], input.Shape[3], lay.inH, lay.inW)
	}
	if lay.batch <= 0 {
		return lay, fmt.Errorf("cnn2: invalid batch")
	}
	want := lay.batch * lay.inC * lay.inH * lay.inW
	if len(input.Data) < want {
		return lay, fmt.Errorf("cnn2: data len %d < %d", len(input.Data), want)
	}
	return lay, nil
}

func spatialN(lay layout) int { return lay.outH * lay.outW }

// im2col → [batch*outH*outW, inC*k*k]
func im2col[T core.Numeric](input *core.Tensor[T], lay layout) *core.Tensor[T] {
	cols := lay.inC * lay.kSize * lay.kSize
	rows := lay.batch * spatialN(lay)
	out := core.NewTensor[T](rows, cols)
	for b := 0; b < lay.batch; b++ {
		for oh := 0; oh < lay.outH; oh++ {
			for ow := 0; ow < lay.outW; ow++ {
				base := (b*spatialN(lay) + oh*lay.outW + ow) * cols
				for ic := 0; ic < lay.inC; ic++ {
					for kh := 0; kh < lay.kSize; kh++ {
						for kw := 0; kw < lay.kSize; kw++ {
							ih := oh*lay.stride + kh - lay.padding
							iw := ow*lay.stride + kw - lay.padding
							var v T
							if ih >= 0 && ih < lay.inH && iw >= 0 && iw < lay.inW {
								v = input.Data[b*lay.inC*lay.inH*lay.inW+ic*lay.inH*lay.inW+ih*lay.inW+iw]
							}
							out.Data[base+ic*lay.kSize*lay.kSize+kh*lay.kSize+kw] = v
						}
					}
				}
			}
		}
	}
	return out
}

// col2im scatters → [batch, inC, H, W].
func col2im[T core.Numeric](gxCol *core.Tensor[T], lay layout) *core.Tensor[T] {
	out := core.NewTensor[T](lay.batch, lay.inC, lay.inH, lay.inW)
	acc := make([]float64, lay.batch*lay.inC*lay.inH*lay.inW)
	cols := lay.inC * lay.kSize * lay.kSize
	for b := 0; b < lay.batch; b++ {
		for oh := 0; oh < lay.outH; oh++ {
			for ow := 0; ow < lay.outW; ow++ {
				base := (b*spatialN(lay) + oh*lay.outW + ow) * cols
				for ic := 0; ic < lay.inC; ic++ {
					for kh := 0; kh < lay.kSize; kh++ {
						for kw := 0; kw < lay.kSize; kw++ {
							ih := oh*lay.stride + kh - lay.padding
							iw := ow*lay.stride + kw - lay.padding
							if ih < 0 || ih >= lay.inH || iw < 0 || iw >= lay.inW {
								continue
							}
							acc[b*lay.inC*lay.inH*lay.inW+ic*lay.inH*lay.inW+ih*lay.inW+iw] +=
								core.AsFloat64(gxCol.Data[base+ic*lay.kSize*lay.kSize+kh*lay.kSize+kw])
						}
					}
				}
			}
		}
	}
	for i := range out.Data {
		out.Data[i] = core.FromFloat64[T](acc[i])
	}
	return out
}

// dense [batch*outH*outW, filters] → loom [batch, filters, outH, outW]
func loomFromDense[T core.Numeric](flat []T, batch, filters, outH, outW int) *core.Tensor[T] {
	out := core.NewTensor[T](batch, filters, outH, outW)
	sp := outH * outW
	for b := 0; b < batch; b++ {
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				for f := 0; f < filters; f++ {
					out.Data[b*filters*sp+f*sp+oh*outW+ow] = flat[(b*sp+oh*outW+ow)*filters+f]
				}
			}
		}
	}
	return out
}

// loom [batch, filters, outH, outW] → dense [batch*outH*outW, filters]
func denseFromLoom[T core.Numeric](loom []T, batch, filters, outH, outW int) *core.Tensor[T] {
	sp := outH * outW
	out := core.NewTensor[T](batch*sp, filters)
	for b := 0; b < batch; b++ {
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				for f := 0; f < filters; f++ {
					out.Data[(b*sp+oh*outW+ow)*filters+f] = loom[b*filters*sp+f*sp+oh*outW+ow]
				}
			}
		}
	}
	return out
}
