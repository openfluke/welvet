package quant

import (
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"sync"
)

// BinaryG128Group is Bonsai / MLX 1-bit group width (FP16 scale per 128 weights).
const BinaryG128Group = 128

// BlobFromMLX1Bit builds a BinaryPacked blob from MLX AffineQuantized 1-bit tensors.
//
// MLX layout (per row):
//   weight U32 [cols/32] — 1 bit/weight, LSB-first within each word
//   scales F16 [cols/128], biases F16 [cols/128]
// Bonsai packs scale-only ±s_g as s_mlx=2·s_g, bias=−s_g.
// We store s_g in Scales and drop biases (w = ±s_g).
//
// Row-aligned groups: Scales[row*(cols/128) + g], Raw holds little-endian u32 words.
func BlobFromMLX1Bit(weightU32 []uint32, scalesF16, biasesF16 []float32, rows, cols int) (*Blob, error) {
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("BlobFromMLX1Bit: bad shape %dx%d", rows, cols)
	}
	if cols%32 != 0 {
		return nil, fmt.Errorf("BlobFromMLX1Bit: cols %d not multiple of 32", cols)
	}
	wordsPerRow := cols / 32
	groupsPerRow := cols / BinaryG128Group
	if groupsPerRow == 0 || cols%BinaryG128Group != 0 {
		return nil, fmt.Errorf("BlobFromMLX1Bit: cols %d not multiple of %d", cols, BinaryG128Group)
	}
	if len(weightU32) < rows*wordsPerRow {
		return nil, fmt.Errorf("BlobFromMLX1Bit: weight short %d need %d", len(weightU32), rows*wordsPerRow)
	}
	if len(scalesF16) < rows*groupsPerRow || len(biasesF16) < rows*groupsPerRow {
		return nil, fmt.Errorf("BlobFromMLX1Bit: scales/biases short")
	}

	scales := make([]float32, rows*groupsPerRow)
	raw := make([]byte, rows*wordsPerRow*4)
	for r := 0; r < rows; r++ {
		for g := 0; g < groupsPerRow; g++ {
			si := r*groupsPerRow + g
			sg := -biasesF16[si]
			if sg == 0 {
				sg = scalesF16[si] * 0.5
			}
			if sg == 0 {
				sg = 1
			}
			scales[si] = sg
		}
		for w := 0; w < wordsPerRow; w++ {
			off := (r*wordsPerRow + w) * 4
			binary.LittleEndian.PutUint32(raw[off:], weightU32[r*wordsPerRow+w])
		}
	}
	return &Blob{
		Format:       FormatBinaryPacked,
		Rows:         rows,
		Cols:         cols,
		Raw:          raw,
		Scales:       scales,
		BlockWeights: BinaryG128Group,
	}, nil
}

func isBinaryG128(b *Blob) bool {
	return b != nil && b.Format == FormatBinaryPacked && b.BlockWeights == BinaryG128Group
}

// IsBinaryG128 reports MLX/Bonsai g128 binary layout.
func IsBinaryG128(b *Blob) bool { return isBinaryG128(b) }

// InferBinaryBlockWeights sets BlockWeights from Scales layout after ENTITY load.
func InferBinaryBlockWeights(b *Blob) {
	if b == nil || b.Format != FormatBinaryPacked {
		return
	}
	if b.BlockWeights == BinaryG128Group || b.BlockWeights == binaryGroup {
		return
	}
	if b.Cols > 0 && b.Cols%BinaryG128Group == 0 {
		need := b.Rows * (b.Cols / BinaryG128Group)
		if len(b.Scales) == need {
			b.BlockWeights = BinaryG128Group
			return
		}
	}
	b.BlockWeights = binaryGroup
}

func forEachBinaryG128(b *Blob, fn func(i int, w float32)) {
	rows, cols := b.Rows, b.Cols
	groupsPerRow := cols / BinaryG128Group
	wordsPerRow := cols / 32
	for r := 0; r < rows; r++ {
		for g := 0; g < groupsPerRow; g++ {
			scale := float32(1)
			si := r*groupsPerRow + g
			if si < len(b.Scales) {
				scale = b.Scales[si]
			}
			for wi := 0; wi < 4; wi++ {
				wordOff := (r*wordsPerRow + g*4 + wi) * 4
				if wordOff+4 > len(b.Raw) {
					return
				}
				word := getU32(b.Raw[wordOff:])
				base := r*cols + g*BinaryG128Group + wi*32
				for j := 0; j < 32; j++ {
					i := base + j
					if i >= rows*cols {
						return
					}
					if (word>>uint(j))&1 != 0 {
						fn(i, scale)
					} else {
						fn(i, -scale)
					}
				}
			}
		}
	}
}

func decodeRowBinaryG128(b *Blob, row int, dst []float32) error {
	cols := b.Cols
	if row < 0 || row >= b.Rows || len(dst) < cols {
		return fmt.Errorf("decodeRowBinaryG128: bad row/dst")
	}
	groupsPerRow := cols / BinaryG128Group
	wordsPerRow := cols / 32
	for g := 0; g < groupsPerRow; g++ {
		scale := float32(1)
		si := row*groupsPerRow + g
		if si < len(b.Scales) {
			scale = b.Scales[si]
		}
		for wi := 0; wi < 4; wi++ {
			wordOff := (row*wordsPerRow + g*4 + wi) * 4
			word := getU32(b.Raw[wordOff:])
			base := g*BinaryG128Group + wi*32
			for j := 0; j < 32; j++ {
				if (word>>uint(j))&1 != 0 {
					dst[base+j] = scale
				} else {
					dst[base+j] = -scale
				}
			}
		}
	}
	return nil
}

func matVecBinaryG128(b *Blob, x, y []float32) error {
	rows, cols := b.Rows, b.Cols
	if len(x) < cols || len(y) < rows {
		return fmt.Errorf("matVecBinaryG128: shape")
	}
	if rows < 64 || runtime.NumCPU() < 2 {
		matVecBinaryG128Rows(b, x, y, 0, rows)
		return nil
	}
	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = runtime.NumCPU()
	}
	if workers > rows {
		workers = rows
	}
	chunk := (rows + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo := w * chunk
		hi := lo + chunk
		if hi > rows {
			hi = rows
		}
		if lo >= hi {
			break
		}
		wg.Add(1)
		go func(r0, r1 int) {
			defer wg.Done()
			matVecBinaryG128Rows(b, x, y, r0, r1)
		}(lo, hi)
	}
	wg.Wait()
	return nil
}

func matVecBinaryG128Rows(b *Blob, x, y []float32, rowLo, rowHi int) {
	cols := b.Cols
	groupsPerRow := cols / BinaryG128Group
	wordsPerRow := cols / 32
	raw := b.Raw
	scales := b.Scales
	for r := rowLo; r < rowHi; r++ {
		var acc float32
		rowWords := r * wordsPerRow
		rowScale := r * groupsPerRow
		for g := 0; g < groupsPerRow; g++ {
			scale := float32(1)
			si := rowScale + g
			if si < len(scales) {
				scale = scales[si]
			}
			baseWord := rowWords + g*4
			colBase := g * BinaryG128Group
			for wi := 0; wi < 4; wi++ {
				word := getU32(raw[(baseWord+wi)*4:])
				base := colBase + wi*32
				// ±scale * x[j] — pull scale out of the bit loop
				var s float32
				for j := 0; j < 32; j++ {
					v := x[base+j]
					if (word>>uint(j))&1 != 0 {
						s += v
					} else {
						s -= v
					}
				}
				acc += scale * s
			}
		}
		y[r] = acc
	}
}

// F16BitsToFloat32 converts IEEE754 binary16 bits to float32.
func F16BitsToFloat32(u uint16) float32 {
	sign := (u >> 15) & 1
	exp := (u >> 10) & 0x1f
	mant := u & 0x3ff
	var val float32
	switch {
	case exp == 0:
		if mant == 0 {
			val = 0
		} else {
			val = float32(math.Ldexp(float64(mant)/1024.0, -14))
		}
	case exp == 31:
		if mant == 0 {
			val = float32(math.Inf(1))
		} else {
			val = float32(math.NaN())
		}
	default:
		val = float32(math.Ldexp(1.0+float64(mant)/1024.0, int(exp)-15))
	}
	if sign != 0 {
		return -val
	}
	return val
}
