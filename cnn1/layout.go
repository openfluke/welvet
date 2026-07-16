package cnn1

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
		return lay, fmt.Errorf("cnn1: empty input")
	}
	if len(input.Shape) != 3 {
		return lay, fmt.Errorf("cnn1: shape need [batch,inChannels,seqLen], got %v", input.Shape)
	}
	lay.batch = input.Shape[0]
	if input.Shape[1] != lay.inC {
		return lay, fmt.Errorf("cnn1: channels %d != %d", input.Shape[1], lay.inC)
	}
	if input.Shape[2] != lay.seq {
		return lay, fmt.Errorf("cnn1: seq %d != %d", input.Shape[2], lay.seq)
	}
	if lay.batch <= 0 {
		return lay, fmt.Errorf("cnn1: invalid batch")
	}
	want := lay.batch * lay.inC * lay.seq
	if len(input.Data) < want {
		return lay, fmt.Errorf("cnn1: data len %d < %d", len(input.Data), want)
	}
	return lay, nil
}

// im2col → [batch*outLen, inC*k]
func im2col[T core.Numeric](input *core.Tensor[T], lay layout) *core.Tensor[T] {
	cols := lay.inC * lay.kSize
	rows := lay.batch * lay.outLen
	out := core.NewTensor[T](rows, cols)
	for b := 0; b < lay.batch; b++ {
		for o := 0; o < lay.outLen; o++ {
			base := (b*lay.outLen + o) * cols
			for ic := 0; ic < lay.inC; ic++ {
				for k := 0; k < lay.kSize; k++ {
					inPos := o*lay.stride + k - lay.padding
					var v T
					if inPos >= 0 && inPos < lay.seq {
						v = input.Data[b*lay.inC*lay.seq+ic*lay.seq+inPos]
					}
					out.Data[base+ic*lay.kSize+k] = v
				}
			}
		}
	}
	return out
}

// col2im scatters [batch*outLen, inC*k] → [batch, inC, seq].
func col2im[T core.Numeric](gxCol *core.Tensor[T], lay layout) *core.Tensor[T] {
	out := core.NewTensor[T](lay.batch, lay.inC, lay.seq)
	acc := make([]float64, lay.batch*lay.inC*lay.seq)
	cols := lay.inC * lay.kSize
	for b := 0; b < lay.batch; b++ {
		for o := 0; o < lay.outLen; o++ {
			base := (b*lay.outLen + o) * cols
			for ic := 0; ic < lay.inC; ic++ {
				for k := 0; k < lay.kSize; k++ {
					inPos := o*lay.stride + k - lay.padding
					if inPos < 0 || inPos >= lay.seq {
						continue
					}
					acc[b*lay.inC*lay.seq+ic*lay.seq+inPos] += core.AsFloat64(gxCol.Data[base+ic*lay.kSize+k])
				}
			}
		}
	}
	for i := range out.Data {
		out.Data[i] = core.FromFloat64[T](acc[i])
	}
	return out
}

// dense [batch*outLen, filters] → loom [batch, filters, outLen]
func loomFromDense[T core.Numeric](flat []T, batch, filters, outLen int) *core.Tensor[T] {
	out := core.NewTensor[T](batch, filters, outLen)
	for b := 0; b < batch; b++ {
		for o := 0; o < outLen; o++ {
			for f := 0; f < filters; f++ {
				out.Data[b*filters*outLen+f*outLen+o] = flat[(b*outLen+o)*filters+f]
			}
		}
	}
	return out
}

// loom [batch, filters, outLen] → dense flat [batch*outLen, filters]
func denseFromLoom[T core.Numeric](loom []T, batch, filters, outLen int) *core.Tensor[T] {
	out := core.NewTensor[T](batch*outLen, filters)
	for b := 0; b < batch; b++ {
		for o := 0; o < outLen; o++ {
			for f := 0; f < filters; f++ {
				out.Data[(b*outLen+o)*filters+f] = loom[b*filters*outLen+f*outLen+o]
			}
		}
	}
	return out
}
