package dense

import (
	"fmt"

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
		return forwardSIMDTernaryPacked(l, x, y, batch, in, out)
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
	quant.EnsureQ8SIMDCache(b)
	if len(b.Scales) == 0 || len(b.Int8QS) < out*in {
		return fmt.Errorf("dense: Q8_0 SIMD cache missing")
	}
	for bat := 0; bat < batch; bat++ {
		xRow := core.SliceAsFloat32(x[bat*in : (bat+1)*in])
		writeGemvF32(y[bat*out:(bat+1)*out], out, func(dst []float32) {
			gemvQ8ParallelF32(b.Scales, b.Int8QS, xRow, dst, out, in)
		})
	}
	return nil
}

func forwardSIMDBinary[T core.Numeric](l *Layer, x, y []T, batch, in, out int) error {
	b := l.Weights.Packed
	if b == nil || b.Format != quant.FormatBinaryPacked {
		return fmt.Errorf("dense: BinaryPacked missing")
	}
	for bat := 0; bat < batch; bat++ {
		xRow := core.SliceAsFloat32(x[bat*in : (bat+1)*in])
		writeGemvF32(y[bat*out:(bat+1)*out], out, func(dst []float32) {
			_ = matVecBitNetF32(b, xRow, dst)
		})
	}
	return nil
}

func forwardSIMDTernaryPacked[T core.Numeric](l *Layer, x, y []T, batch, in, out int) error {
	b := l.Weights.Packed
	if b == nil || b.Format != quant.FormatTernaryPacked {
		return fmt.Errorf("dense: TernaryPacked missing")
	}
	for bat := 0; bat < batch; bat++ {
		xRow := core.SliceAsFloat32(x[bat*in : (bat+1)*in])
		writeGemvF32(y[bat*out:(bat+1)*out], out, func(dst []float32) {
			_ = matVecBitNetF32(b, xRow, dst)
		})
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
