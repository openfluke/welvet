package dense

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
)

// forwardSIMDBlockFused — classic Q4_1/Q5 + k/IQ via fused Dot* (Int8QS + scales; no F32 inflate).
func forwardSIMDBlockFused[T core.Numeric](l *Layer, x, y []T, batch, in, out int) error {
	b := l.Weights.Packed
	if b == nil {
		return fmt.Errorf("dense: block-fused missing blob")
	}
	quant.EnsureFusedSIMDCache(b)
	for bat := 0; bat < batch; bat++ {
		xRow := core.SliceAsFloat32(x[bat*in : (bat+1)*in])
		var err error
		writeGemvF32(y[bat*out:(bat+1)*out], out, func(dst []float32) {
			switch {
			case b.Format == quant.FormatQ4_1:
				err = matVecQ4_1Fused(b, xRow, dst)
			case b.Format == quant.FormatQ5_0:
				err = matVecQ5_0Fused(b, xRow, dst)
			case b.Format == quant.FormatQ5_1:
				err = matVecQ5_1Fused(b, xRow, dst)
			case b.Format.IsKQuant():
				err = matVecKSIMD(b, xRow, dst)
			case isIQ(b.Format):
				err = matVecIQSIMD(b, xRow, dst)
			default:
				err = fmt.Errorf("dense: no block-fused path for %s", b.Format)
			}
		})
		if err != nil {
			return err
		}
	}
	return nil
}

func isIQ(f quant.Format) bool {
	switch f {
	case quant.FormatIQ1_S, quant.FormatIQ2_XXS, quant.FormatIQ2_XS,
		quant.FormatIQ3_XXS, quant.FormatIQ3_S, quant.FormatIQ4_NL, quant.FormatIQ4_XS:
		return true
	default:
		return false
	}
}

func matVecQ4_1Fused(b *quant.Blob, x, y []float32) error {
	if len(b.Scales) == 0 || len(b.Mins) == 0 || len(b.Q4Packed) == 0 {
		return fmt.Errorf("dense: Q4_1 SIMD cache missing")
	}
	in, out := b.Cols, b.Rows
	gemvQ41ParallelF32(b.Scales, b.Mins, b.Q4Packed, x, y, out, in)
	return nil
}

func matVecQ5_0Fused(b *quant.Blob, x, y []float32) error {
	if len(b.Scales) == 0 || len(b.Int8QS) < b.Rows*b.Cols {
		return fmt.Errorf("dense: Q5_0 SIMD cache missing")
	}
	in, out := b.Cols, b.Rows
	gemvQ5_0ParallelF32(b.Scales, b.Int8QS, x, y, out, in)
	return nil
}

func matVecQ5_1Fused(b *quant.Blob, x, y []float32) error {
	if len(b.Scales) == 0 || len(b.Mins) == 0 || len(b.Int8QS) < b.Rows*b.Cols {
		return fmt.Errorf("dense: Q5_1 SIMD cache missing")
	}
	in, out := b.Cols, b.Rows
	gemvQ5_1ParallelF32(b.Scales, b.Mins, b.Int8QS, x, y, out, in)
	return nil
}
