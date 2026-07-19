package dense

import (
	"fmt"

	"github.com/openfluke/welvet/quant"
)

// matVecKSIMD — once-inflated FP32 cache + parallel DotTile (Lucy-class GEMV schedule).
// Packed Raw stays; F32Cache is the simd_fuse compute view until true k-quant .s lands.
func matVecKSIMD(b *quant.Blob, x, y []float32) error {
	quant.EnsureFloatCache(b)
	in, out := b.Cols, b.Rows
	if len(b.F32Cache) < out*in {
		return fmt.Errorf("dense: k-quant F32Cache missing for %s", b.Format)
	}
	if len(x) < in || len(y) < out {
		return fmt.Errorf("dense: k-quant matvec shape")
	}
	gemvF32ParallelF32(b.F32Cache, x, y, out, in)
	return nil
}

// matVecIQSIMD — same inflate-once + parallel DotTile path as k-quant.
func matVecIQSIMD(b *quant.Blob, x, y []float32) error {
	quant.EnsureFloatCache(b)
	in, out := b.Cols, b.Rows
	if len(b.F32Cache) < out*in {
		return fmt.Errorf("dense: IQ F32Cache missing for %s", b.Format)
	}
	if len(x) < in || len(y) < out {
		return fmt.Errorf("dense: IQ matvec shape")
	}
	gemvF32ParallelF32(b.F32Cache, x, y, out, in)
	return nil
}
