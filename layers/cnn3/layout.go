package cnn3

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
		return lay, fmt.Errorf("cnn3: empty input")
	}
	if len(input.Shape) != 5 {
		return lay, fmt.Errorf("cnn3: shape need [batch,inChannels,D,H,W], got %v", input.Shape)
	}
	lay.batch = input.Shape[0]
	if input.Shape[1] != lay.inC {
		return lay, fmt.Errorf("cnn3: channels %d != %d", input.Shape[1], lay.inC)
	}
	if input.Shape[2] != lay.inD || input.Shape[3] != lay.inH || input.Shape[4] != lay.inW {
		return lay, fmt.Errorf("cnn3: spatial %dx%dx%d != %dx%dx%d",
			input.Shape[2], input.Shape[3], input.Shape[4], lay.inD, lay.inH, lay.inW)
	}
	if lay.batch <= 0 {
		return lay, fmt.Errorf("cnn3: invalid batch")
	}
	want := lay.batch * lay.inC * lay.inD * lay.inH * lay.inW
	if len(input.Data) < want {
		return lay, fmt.Errorf("cnn3: data len %d < %d", len(input.Data), want)
	}
	return lay, nil
}

func spatialN(lay layout) int { return lay.outD * lay.outH * lay.outW }

func inIdx(lay layout, b, ic, id, ih, iw int) int {
	return b*lay.inC*lay.inD*lay.inH*lay.inW + ic*lay.inD*lay.inH*lay.inW + id*lay.inH*lay.inW + ih*lay.inW + iw
}

func patchOff(lay layout, ic, kd, kh, kw int) int {
	return ic*lay.kSize*lay.kSize*lay.kSize + kd*lay.kSize*lay.kSize + kh*lay.kSize + kw
}

// im2col → [batch*outD*outH*outW, inC*k³]
func im2col[T core.Numeric](input *core.Tensor[T], lay layout) *core.Tensor[T] {
	cols := lay.inC * lay.kSize * lay.kSize * lay.kSize
	rows := lay.batch * spatialN(lay)
	out := core.NewTensor[T](rows, cols)
	for b := 0; b < lay.batch; b++ {
		for od := 0; od < lay.outD; od++ {
			for oh := 0; oh < lay.outH; oh++ {
				for ow := 0; ow < lay.outW; ow++ {
					base := (b*spatialN(lay) + od*lay.outH*lay.outW + oh*lay.outW + ow) * cols
					for ic := 0; ic < lay.inC; ic++ {
						for kd := 0; kd < lay.kSize; kd++ {
							for kh := 0; kh < lay.kSize; kh++ {
								for kw := 0; kw < lay.kSize; kw++ {
									id := od*lay.stride + kd - lay.padding
									ih := oh*lay.stride + kh - lay.padding
									iw := ow*lay.stride + kw - lay.padding
									var v T
									if id >= 0 && id < lay.inD && ih >= 0 && ih < lay.inH && iw >= 0 && iw < lay.inW {
										v = input.Data[inIdx(lay, b, ic, id, ih, iw)]
									}
									out.Data[base+patchOff(lay, ic, kd, kh, kw)] = v
								}
							}
						}
					}
				}
			}
		}
	}
	return out
}

// col2im scatters → [batch, inC, D, H, W].
func col2im[T core.Numeric](gxCol *core.Tensor[T], lay layout) *core.Tensor[T] {
	out := core.NewTensor[T](lay.batch, lay.inC, lay.inD, lay.inH, lay.inW)
	acc := make([]float64, lay.batch*lay.inC*lay.inD*lay.inH*lay.inW)
	cols := lay.inC * lay.kSize * lay.kSize * lay.kSize
	for b := 0; b < lay.batch; b++ {
		for od := 0; od < lay.outD; od++ {
			for oh := 0; oh < lay.outH; oh++ {
				for ow := 0; ow < lay.outW; ow++ {
					base := (b*spatialN(lay) + od*lay.outH*lay.outW + oh*lay.outW + ow) * cols
					for ic := 0; ic < lay.inC; ic++ {
						for kd := 0; kd < lay.kSize; kd++ {
							for kh := 0; kh < lay.kSize; kh++ {
								for kw := 0; kw < lay.kSize; kw++ {
									id := od*lay.stride + kd - lay.padding
									ih := oh*lay.stride + kh - lay.padding
									iw := ow*lay.stride + kw - lay.padding
									if id < 0 || id >= lay.inD || ih < 0 || ih >= lay.inH || iw < 0 || iw >= lay.inW {
										continue
									}
									acc[inIdx(lay, b, ic, id, ih, iw)] +=
										core.AsFloat64(gxCol.Data[base+patchOff(lay, ic, kd, kh, kw)])
								}
							}
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

// dense [batch*sp, filters] → loom [batch, filters, outD, outH, outW]
func loomFromDense[T core.Numeric](flat []T, batch, filters, outD, outH, outW int) *core.Tensor[T] {
	out := core.NewTensor[T](batch, filters, outD, outH, outW)
	sp := outD * outH * outW
	for b := 0; b < batch; b++ {
		for od := 0; od < outD; od++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					for f := 0; f < filters; f++ {
						si := od*outH*outW + oh*outW + ow
						out.Data[b*filters*sp+f*sp+si] = flat[(b*sp+si)*filters+f]
					}
				}
			}
		}
	}
	return out
}

// loom → dense [batch*sp, filters]
func denseFromLoom[T core.Numeric](loom []T, batch, filters, outD, outH, outW int) *core.Tensor[T] {
	sp := outD * outH * outW
	out := core.NewTensor[T](batch*sp, filters)
	for b := 0; b < batch; b++ {
		for od := 0; od < outD; od++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					for f := 0; f < filters; f++ {
						si := od*outH*outW + oh*outW + ow
						out.Data[(b*sp+si)*filters+f] = loom[b*filters*sp+f*sp+si]
					}
				}
			}
		}
	}
	return out
}
