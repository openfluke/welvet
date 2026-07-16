package dense

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/simd"
)

// forwardSIMDBlockFused — k-quant / IQ / Q4_1 / Q5_*: decode groups into scratch + DotTile.
// Never unpacks the full matrix.
func forwardSIMDBlockFused[T core.Numeric](l *Layer, x, y []T, batch, in, out int) error {
	b := l.Weights.Packed
	if b == nil {
		return fmt.Errorf("dense: block-fused missing blob")
	}
	for bat := 0; bat < batch; bat++ {
		xRow := core.SliceAsFloat32(x[bat*in : (bat+1)*in])
		yF := make([]float32, out)
		var err error
		switch {
		case b.Format.IsKQuant():
			err = matVecKSIMD(b, xRow, yF)
		case isIQ(b.Format):
			err = matVecIQSIMD(b, xRow, yF)
		case b.Format == quant.FormatQ4_1:
			err = matVecQ4_1SIMD(b, xRow, yF)
		case b.Format == quant.FormatQ5_0:
			err = matVecQ5_0SIMD(b, xRow, yF)
		case b.Format == quant.FormatQ5_1:
			err = matVecQ5_1SIMD(b, xRow, yF)
		default:
			return fmt.Errorf("dense: no block-fused path for %s", b.Format)
		}
		if err != nil {
			return err
		}
		core.SliceFromFloat32(yF, y[bat*out:(bat+1)*out])
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

func matVecKSIMD(b *quant.Blob, x, y []float32) error {
	for i := range y {
		y[i] = 0
	}
	scratch := make([]float32, 16)
	var curRow, curCol int
	nFill := 0
	flush := func() {
		if nFill == 0 {
			return
		}
		y[curRow] += float32(simd.DotTile(x[curCol:curCol+nFill], scratch[:nFill], 0, nFill, 0))
		nFill = 0
	}
	err := quant.ForEachK(b, func(i int, w float32) {
		r, c := i/b.Cols, i%b.Cols
		if nFill > 0 && (r != curRow || c != curCol+nFill) {
			flush()
		}
		if nFill == 0 {
			curRow, curCol = r, c
		}
		scratch[nFill] = w
		nFill++
		if nFill == 16 {
			flush()
		}
	})
	flush()
	return err
}

func matVecIQSIMD(b *quant.Blob, x, y []float32) error {
	for i := range y {
		y[i] = 0
	}
	scratch := make([]float32, 32)
	var curRow, curCol int
	nFill := 0
	flush := func() {
		if nFill == 0 {
			return
		}
		y[curRow] += float32(simd.DotTile(x[curCol:curCol+nFill], scratch[:nFill], 0, nFill, 0))
		nFill = 0
	}
	err := quant.ForEachIQ(b, func(i int, w float32) {
		r, c := i/b.Cols, i%b.Cols
		if nFill > 0 && (r != curRow || c != curCol+nFill) {
			flush()
		}
		if nFill == 0 {
			curRow, curCol = r, c
		}
		scratch[nFill] = w
		nFill++
		if nFill == 32 {
			flush()
		}
	})
	flush()
	return err
}

func matVecQ4_1SIMD(b *quant.Blob, x, y []float32) error {
	return matVecBlock32SIMD(b, x, y, decodeQ4_1Block)
}

func matVecQ5_0SIMD(b *quant.Blob, x, y []float32) error {
	return matVecBlock32SIMD(b, x, y, decodeQ5_0Block)
}

func matVecQ5_1SIMD(b *quant.Blob, x, y []float32) error {
	return matVecBlock32SIMD(b, x, y, decodeQ5_1Block)
}

type blockDecoder func(b *quant.Blob, bi int, dst []float32) (start int, n int, ok bool)

func matVecBlock32SIMD(b *quant.Blob, x, y []float32, dec blockDecoder) error {
	for i := range y {
		y[i] = 0
	}
	scratch := make([]float32, 32)
	n := b.Rows * b.Cols
	blocks := (n + 31) / 32
	for bi := 0; bi < blocks; bi++ {
		start, nn, ok := dec(b, bi, scratch)
		if !ok || nn == 0 {
			continue
		}
		// Flat index start → may span multiple rows; process contiguous runs.
		i0 := start
		for i0 < start+nn {
			r := i0 / b.Cols
			c := i0 % b.Cols
			run := b.Cols - c
			if i0+run > start+nn {
				run = start + nn - i0
			}
			off := i0 - start
			y[r] += float32(simd.DotTile(x[c:c+run], scratch[off:off+run], 0, run, 0))
			i0 += run
		}
	}
	return nil
}

func decodeQ4_1Block(b *quant.Blob, bi int, dst []float32) (int, int, bool) {
	const bw, bb = 32, 24
	if len(b.Raw) < (bi+1)*bb {
		return 0, 0, false
	}
	off := bi * bb
	scale := quant.GetF32(b.Raw[off:])
	mn := quant.GetF32(b.Raw[off+4:])
	start := bi * bw
	n := b.Rows * b.Cols
	nn := bw
	if start+nn > n {
		nn = n - start
	}
	for j := 0; j < 16 && j*2 < nn; j++ {
		w := b.Raw[off+8+j]
		for k := 0; k < 2 && j*2+k < nn; k++ {
			q := int((w >> (4 * k)) & 0xF)
			dst[j*2+k] = mn + float32(q)*scale
		}
	}
	return start, nn, true
}

func decodeQ5_0Block(b *quant.Blob, bi int, dst []float32) (int, int, bool) {
	const bw, bb = 32, 24
	if len(b.Raw) < (bi+1)*bb {
		return 0, 0, false
	}
	off := bi * bb
	scale := quant.GetF32(b.Raw[off:])
	start := bi * bw
	n := b.Rows * b.Cols
	nn := bw
	if start+nn > n {
		nn = n - start
	}
	br := quant.NewBitReader(b.Raw[off+4 : off+bb])
	for j := 0; j < nn; j++ {
		q := br.Read(5)
		dst[j] = float32(int(q)-16) * scale
	}
	return start, nn, true
}

func decodeQ5_1Block(b *quant.Blob, bi int, dst []float32) (int, int, bool) {
	const bw, bb = 32, 28
	if len(b.Raw) < (bi+1)*bb {
		return 0, 0, false
	}
	off := bi * bb
	scale := quant.GetF32(b.Raw[off:])
	mn := quant.GetF32(b.Raw[off+4:])
	start := bi * bw
	n := b.Rows * b.Cols
	nn := bw
	if start+nn > n {
		nn = n - start
	}
	br := quant.NewBitReader(b.Raw[off+8 : off+bb])
	for j := 0; j < nn; j++ {
		q := br.Read(5)
		dst[j] = mn + float32(q)*scale
	}
	return start, nn, true
}
