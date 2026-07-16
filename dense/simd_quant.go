package dense

import (
	"fmt"
	"math"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/simd"
)

// forwardSIMDPacked dispatches fused / stream-fused kernels for Format != None.
func forwardSIMDPacked[T core.Numeric](l *Layer, x, y []T, batch, in, out int) error {
	switch l.Weights.Format {
	case quant.FormatQ4_0:
		return forwardSIMDQ4_0(l, x, y, batch, in, out)
	case quant.FormatQ8_0:
		return forwardSIMDQ8_0(l, x, y, batch, in, out)
	case quant.FormatTernaryPacked:
		return forwardSIMDTernary(l, x, y, batch, in, out)
	case quant.FormatBinaryPacked:
		return forwardSIMDBinary(l, x, y, batch, in, out)
	case quant.FormatQ4_1, quant.FormatQ5_0, quant.FormatQ5_1:
		return forwardSIMDBlockFused(l, x, y, batch, in, out)
	case quant.FormatQ2_K, quant.FormatQ3_K, quant.FormatQ4_K, quant.FormatQ5_K, quant.FormatQ6_K:
		return forwardSIMDBlockFused(l, x, y, batch, in, out)
	case quant.FormatIQ1_S, quant.FormatIQ2_XXS, quant.FormatIQ2_XS,
		quant.FormatIQ3_XXS, quant.FormatIQ3_S, quant.FormatIQ4_NL, quant.FormatIQ4_XS:
		return forwardSIMDBlockFused(l, x, y, batch, in, out)
	default:
		return fmt.Errorf("dense: SIMD packed unsupported format %s", l.Weights.Format)
	}
}

func forwardSIMDQ8_0[T core.Numeric](l *Layer, x, y []T, batch, in, out int) error {
	b := l.Weights.Packed
	if b == nil || b.Format != quant.FormatQ8_0 {
		return fmt.Errorf("dense: Q8_0 packed missing")
	}
	const bw, bb = 32, 36
	n := out * in
	blocks := (n + bw - 1) / bw
	if len(b.Raw) < blocks*bb {
		return fmt.Errorf("dense: Q8_0 raw short")
	}
	xq := make([]int8, in)
	wBlk := make([]int8, bw)
	for bat := 0; bat < batch; bat++ {
		xF := core.SliceAsFloat32(x[bat*in : (bat+1)*in])
		actScale := quantizeActsI8(xF, xq)
		yF := make([]float32, out)
		for o := 0; o < out; o++ {
			sum := 0.0
			if in%bw == 0 {
				for c0 := 0; c0 < in; c0 += bw {
					ii := o*in + c0
					boff := (ii / bw) * bb
					sc := math.Float32frombits(uint32(b.Raw[boff]) | uint32(b.Raw[boff+1])<<8 |
						uint32(b.Raw[boff+2])<<16 | uint32(b.Raw[boff+3])<<24)
					for j := 0; j < bw; j++ {
						wBlk[j] = int8(b.Raw[boff+4+j])
					}
					acc := simd.DotI8Tile(xq[c0:], wBlk, 0, 0, bw, 0)
					sum += float64(acc) * float64(sc) * float64(actScale)
				}
			} else {
				for c := 0; c < in; c++ {
					ii := o*in + c
					boff := (ii / bw) * bb
					sc := math.Float32frombits(uint32(b.Raw[boff]) | uint32(b.Raw[boff+1])<<8 |
						uint32(b.Raw[boff+2])<<16 | uint32(b.Raw[boff+3])<<24)
					q := int8(b.Raw[boff+4+(ii%bw)])
					sum += float64(int32(xq[c])*int32(q)) * float64(sc) * float64(actScale)
				}
			}
			yF[o] = float32(sum)
		}
		core.SliceFromFloat32(yF, y[bat*out:(bat+1)*out])
	}
	return nil
}

func forwardSIMDBinary[T core.Numeric](l *Layer, x, y []T, batch, in, out int) error {
	b := l.Weights.Packed
	if b == nil || b.Format != quant.FormatBinaryPacked {
		return fmt.Errorf("dense: BinaryPacked missing")
	}
	scales, words, ok := binaryBlobToGPU(b)
	if !ok {
		return fmt.Errorf("dense: BinaryPacked projection failed")
	}
	// Flat groups index by absolute weight index; per-row we need words covering o*in..
	for bat := 0; bat < batch; bat++ {
		xRow := core.SliceAsFloat32(x[bat*in : (bat+1)*in])
		yF := make([]float32, out)
		for o := 0; o < out; o++ {
			sum := 0.0
			base := o * in
			for c0 := 0; c0 < in; {
				flat := base + c0
				g := flat / 32
				if g >= len(words) {
					break
				}
				bitOff := flat % 32
				nn := 32 - bitOff
				if c0+nn > in {
					nn = in - c0
				}
				sc := float32(1)
				if g < len(scales) {
					sc = scales[g]
				}
				sum += simd.DotBinaryWordOffset(xRow[c0:], words[g], sc, bitOff, nn)
				c0 += nn
			}
			yF[o] = float32(sum)
		}
		core.SliceFromFloat32(yF, y[bat*out:(bat+1)*out])
	}
	return nil
}

// backwardSIMDPacked — dX via quant.MatVecT (packed); dW via outer product (Saxpy).
func backwardSIMDPacked[T core.Numeric](l *Layer, dPre []float64, input *core.Tensor[T], gradIn, gradW *core.Tensor[T], batch, in, out int) error {
	b := l.Weights.Packed
	if b == nil {
		return fmt.Errorf("dense: packed bwd missing blob")
	}
	dW64 := make([]float64, out*in)
	for bat := 0; bat < batch; bat++ {
		gyF := make([]float32, out)
		for o := 0; o < out; o++ {
			gyF[o] = float32(dPre[bat*out+o])
		}
		gxF := make([]float32, in)
		if err := quant.MatVecT(b, gyF, gxF); err != nil {
			return err
		}
		core.SliceFromFloat32(gxF, gradIn.Data[bat*in:(bat+1)*in])
		x32 := core.SliceAsFloat32(input.Data[bat*in : (bat+1)*in])
		for o := 0; o < out; o++ {
			simd.SaxpyF32AccF64(dW64[o*in:(o+1)*in], dPre[bat*out+o], x32, in)
		}
	}
	core.SliceFromFloat64(dW64, gradW.Data)
	return nil
}
