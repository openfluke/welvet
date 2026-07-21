package dense

import (
	"fmt"

	"github.com/openfluke/welvet/quant"
)

// matVecKSIMD — fused group-16 DotKRow over EnsureKSIMDCache (no F32 inflate).
func matVecKSIMD(b *quant.Blob, x, y []float32) error {
	quant.EnsureKSIMDCache(b)
	in, out := b.Cols, b.Rows
	if len(x) < in || len(y) < out {
		return fmt.Errorf("dense: k-quant matvec shape")
	}
	needGroups := (out*in + 15) / 16
	if len(b.Int8QS) < out*in || len(b.Scales) < needGroups {
		return fmt.Errorf("dense: k-quant SIMD cache missing for %s", b.Format)
	}
	hasDmin, mid, ok := quant.KSIMDParams(b)
	if !ok {
		return fmt.Errorf("dense: k-quant SIMD params missing for %s", b.Format)
	}
	if hasDmin && len(b.Mins) < needGroups {
		return fmt.Errorf("dense: k-quant SIMD mins missing for %s", b.Format)
	}
	gemvKParallelF32(b.Scales, b.Mins, b.Int8QS, x, y, out, in, hasDmin, mid)
	return nil
}

// matVecIQSIMD — fused DotIQRow over EnsureIQSIMDCache (no F32 inflate).
func matVecIQSIMD(b *quant.Blob, x, y []float32) error {
	quant.EnsureIQSIMDCache(b)
	in, out := b.Cols, b.Rows
	if len(x) < in || len(y) < out {
		return fmt.Errorf("dense: IQ matvec shape")
	}
	scaleGroup, mid, kind, ok := quant.IQSIMDParams(b)
	if !ok {
		return fmt.Errorf("dense: IQ SIMD params missing for %s", b.Format)
	}
	nScale := (in*out + scaleGroup - 1) / scaleGroup
	if len(b.Int8QS) < out*in || len(b.Scales) < nScale {
		return fmt.Errorf("dense: IQ SIMD cache missing for %s", b.Format)
	}
	gemvIQParallelF32(b.Scales, b.Int8QS, x, y, out, in, scaleGroup, mid, kind)
	return nil
}

// matVecAffineSIMD — fused DotAffineRow over EnsureAffineSIMDCache (no F32 inflate, no fallback).
func matVecAffineSIMD(b *quant.Blob, x, y []float32) error {
	quant.EnsureAffineSIMDCache(b)
	in, out := b.Cols, b.Rows
	if len(x) < in || len(y) < out {
		return fmt.Errorf("dense: Affine matvec shape")
	}
	group := b.BlockWeights
	if group <= 0 {
		group = quant.AffineG64Group
	}
	if in%group != 0 {
		return fmt.Errorf("dense: Affine cols %d not multiple of group %d", in, group)
	}
	gpr := in / group
	if len(b.Int8QS) < out*in || len(b.Scales) < out*gpr || len(b.Mins) < out*gpr {
		return fmt.Errorf("dense: Affine SIMD cache missing")
	}
	gemvAffineParallelF32(b.Scales, b.Mins, b.Int8QS, x, y, out, in, group)
	return nil
}
