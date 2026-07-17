package quant

import (
	"encoding/binary"
	"fmt"
	"runtime"
	"sync"
)

// AffineG64Group is the default MLX AffineQuantized group width (4-bit text encoders).
const AffineG64Group = 64

// BlobFromMLXAffine4 builds an AffinePacked blob from MLX AffineQuantized tensors.
//
// Layout (bits=4, group_size=64 typical):
//
//	weight U32 [rows, cols/8] — 8×4-bit codes per word, LSB-first
//	scales, biases [rows, cols/groupSize] — BF16/F16 decoded to float32
//
// Dequant: w_i = s * code_i + β  with code ∈ [0, 2^bits−1].
// Scales go in Blob.Scales; biases (β) in Blob.Mins. Meta[0] stores bits.
func BlobFromMLXAffine4(weightU32 []uint32, scales, biases []float32, rows, cols, groupSize, bits int) (*Blob, error) {
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("BlobFromMLXAffine4: bad shape %dx%d", rows, cols)
	}
	if groupSize <= 0 {
		groupSize = AffineG64Group
	}
	if bits <= 0 {
		bits = 4
	}
	if bits != 4 {
		return nil, fmt.Errorf("BlobFromMLXAffine4: bits=%d not supported (want 4)", bits)
	}
	codesPerWord := 32 / bits
	if cols%codesPerWord != 0 {
		return nil, fmt.Errorf("BlobFromMLXAffine4: cols %d not multiple of %d", cols, codesPerWord)
	}
	if cols%groupSize != 0 {
		return nil, fmt.Errorf("BlobFromMLXAffine4: cols %d not multiple of group %d", cols, groupSize)
	}
	if groupSize%codesPerWord != 0 {
		return nil, fmt.Errorf("BlobFromMLXAffine4: group %d not multiple of codes/word %d", groupSize, codesPerWord)
	}
	wordsPerRow := cols / codesPerWord
	groupsPerRow := cols / groupSize
	if len(weightU32) < rows*wordsPerRow {
		return nil, fmt.Errorf("BlobFromMLXAffine4: weight short %d need %d", len(weightU32), rows*wordsPerRow)
	}
	needSB := rows * groupsPerRow
	if len(scales) < needSB || len(biases) < needSB {
		return nil, fmt.Errorf("BlobFromMLXAffine4: scales/biases short need %d", needSB)
	}

	sc := make([]float32, needSB)
	mn := make([]float32, needSB)
	copy(sc, scales[:needSB])
	copy(mn, biases[:needSB])
	raw := make([]byte, rows*wordsPerRow*4)
	for r := 0; r < rows; r++ {
		for w := 0; w < wordsPerRow; w++ {
			off := (r*wordsPerRow + w) * 4
			binary.LittleEndian.PutUint32(raw[off:], weightU32[r*wordsPerRow+w])
		}
	}
	return &Blob{
		Format:       FormatAffinePacked,
		Rows:         rows,
		Cols:         cols,
		Raw:          raw,
		Scales:       sc,
		Mins:         mn,
		Meta:         []byte{byte(bits)},
		BlockWeights: groupSize,
	}, nil
}

func affineBits(b *Blob) int {
	if b != nil && len(b.Meta) > 0 && b.Meta[0] > 0 {
		return int(b.Meta[0])
	}
	return 4
}

func isAffineG64(b *Blob) bool {
	return b != nil && b.Format == FormatAffinePacked && b.BlockWeights == AffineG64Group
}

// IsAffinePacked reports MLX AffineQuantized packed layout.
func IsAffinePacked(b *Blob) bool {
	return b != nil && b.Format == FormatAffinePacked
}

// InferAffineBlockWeights sets BlockWeights from Scales layout after ENTITY load.
func InferAffineBlockWeights(b *Blob) {
	if b == nil || b.Format != FormatAffinePacked {
		return
	}
	if b.BlockWeights > 0 && b.Cols%b.BlockWeights == 0 {
		need := b.Rows * (b.Cols / b.BlockWeights)
		if len(b.Scales) == need {
			return
		}
	}
	if b.Cols > 0 && b.Cols%AffineG64Group == 0 {
		need := b.Rows * (b.Cols / AffineG64Group)
		if len(b.Scales) == need {
			b.BlockWeights = AffineG64Group
			return
		}
	}
	// Fall back: infer group from scales length.
	if b.Rows > 0 && len(b.Scales)%b.Rows == 0 {
		gpr := len(b.Scales) / b.Rows
		if gpr > 0 && b.Cols%gpr == 0 {
			b.BlockWeights = b.Cols / gpr
		}
	}
}

func decodeRowAffine(b *Blob, row int, dst []float32) error {
	cols := b.Cols
	if row < 0 || row >= b.Rows || len(dst) < cols {
		return fmt.Errorf("decodeRowAffine: bad row/dst")
	}
	bits := affineBits(b)
	group := b.BlockWeights
	if group <= 0 {
		group = AffineG64Group
	}
	codesPerWord := 32 / bits
	wordsPerRow := cols / codesPerWord
	groupsPerRow := cols / group
	mask := uint32((1 << bits) - 1)
	raw := b.Raw
	scales := b.Scales
	biases := b.Mins
	rowWords := row * wordsPerRow
	rowSB := row * groupsPerRow
	for g := 0; g < groupsPerRow; g++ {
		s := float32(1)
		beta := float32(0)
		si := rowSB + g
		if si < len(scales) {
			s = scales[si]
		}
		if si < len(biases) {
			beta = biases[si]
		}
		colBase := g * group
		wordsInGroup := group / codesPerWord
		for wi := 0; wi < wordsInGroup; wi++ {
			word := getU32(raw[(rowWords+g*wordsInGroup+wi)*4:])
			base := colBase + wi*codesPerWord
			for j := 0; j < codesPerWord; j++ {
				code := (word >> uint(j*bits)) & mask
				dst[base+j] = s*float32(code) + beta
			}
		}
	}
	return nil
}

func unpackAffine(b *Blob) ([]float32, error) {
	if b == nil || b.Format != FormatAffinePacked {
		return nil, errFormat("UnpackAffine", b)
	}
	out := make([]float32, b.Rows*b.Cols)
	for r := 0; r < b.Rows; r++ {
		if err := decodeRowAffine(b, r, out[r*b.Cols:(r+1)*b.Cols]); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func matVecAffine(b *Blob, x, y []float32) error {
	rows, cols := b.Rows, b.Cols
	if len(x) < cols || len(y) < rows {
		return fmt.Errorf("matVecAffine: shape")
	}
	group := b.BlockWeights
	if group <= 0 {
		group = AffineG64Group
	}
	groupsPerRow := cols / group
	// Shared per-group Σx for β·Σx term.
	sumX := make([]float32, groupsPerRow)
	for g := 0; g < groupsPerRow; g++ {
		var s float32
		base := g * group
		for j := 0; j < group; j++ {
			s += x[base+j]
		}
		sumX[g] = s
	}

	if rows < 64 || runtime.NumCPU() < 2 {
		matVecAffineRows(b, x, y, sumX, 0, rows)
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
		go func(rowLo, rowHi int) {
			defer wg.Done()
			matVecAffineRows(b, x, y, sumX, rowLo, rowHi)
		}(lo, hi)
	}
	wg.Wait()
	return nil
}

// matVecAffineRows: y[r] = Σ_j (s·code_j + β)·x[j] for r in [rowLo, rowHi).
func matVecAffineRows(b *Blob, x, y, sumX []float32, rowLo, rowHi int) {
	bits := affineBits(b)
	group := b.BlockWeights
	if group <= 0 {
		group = AffineG64Group
	}
	codesPerWord := 32 / bits
	wordsPerRow := b.Cols / codesPerWord
	groupsPerRow := b.Cols / group
	wordsInGroup := group / codesPerWord
	mask := uint32((1 << bits) - 1)
	raw := b.Raw
	scales := b.Scales
	biases := b.Mins

	for r := rowLo; r < rowHi; r++ {
		var acc float32
		rowWords := r * wordsPerRow
		rowSB := r * groupsPerRow
		for g := 0; g < groupsPerRow; g++ {
			s := scales[rowSB+g]
			beta := biases[rowSB+g]
			colBase := g * group
			baseWord := rowWords + g*wordsInGroup
			var codeDot float32
			for wi := 0; wi < wordsInGroup; wi++ {
				word := getU32(raw[(baseWord+wi)*4:])
				xb := x[colBase+wi*codesPerWord:]
				for j := 0; j < codesPerWord; j++ {
					code := (word >> uint(j*bits)) & mask
					codeDot += float32(code) * xb[j]
				}
			}
			acc += s*codeDot + beta*sumX[g]
		}
		y[r] = acc
	}
}

func matVecTAffine(b *Blob, gy, gx []float32) error {
	// gx += Wᵀ @ gy via row decode (correctness path; slow).
	row := make([]float32, b.Cols)
	for r := 0; r < b.Rows; r++ {
		g := gy[r]
		if g == 0 {
			continue
		}
		if err := decodeRowAffine(b, r, row); err != nil {
			return err
		}
		for j := 0; j < b.Cols; j++ {
			gx[j] += row[j] * g
		}
	}
	return nil
}
