package quant

import "fmt"

// DecodeRow expands one output-row of a packed Blob into dst (len >= Cols).
// Does not allocate a full-matrix unpack for classic / BitNet / binary formats.
func DecodeRow(b *Blob, row int, dst []float32) error {
	if b == nil {
		return errFormat("DecodeRow", b)
	}
	if row < 0 || row >= b.Rows || len(dst) < b.Cols {
		return fmt.Errorf("quant: DecodeRow shape row=%d cols=%d dst=%d", row, b.Cols, len(dst))
	}
	switch b.Format {
	case FormatNone:
		return fmt.Errorf("quant: DecodeRow FormatNone — use weights.DecodeRow")
	case FormatQ8_0:
		return decodeRowQ8_0(b, row, dst)
	case FormatQ4_0:
		return decodeRowQ4_0(b, row, dst)
	case FormatQ4_1:
		return decodeRowViaUnpack(b, row, dst)
	case FormatQ5_0, FormatQ5_1:
		return decodeRowViaUnpack(b, row, dst)
	case FormatQ2_K, FormatQ3_K, FormatQ4_K, FormatQ5_K, FormatQ6_K:
		return decodeRowViaUnpack(b, row, dst)
	case FormatIQ1_S, FormatIQ2_XXS, FormatIQ2_XS, FormatIQ3_XXS, FormatIQ3_S, FormatIQ4_NL, FormatIQ4_XS:
		return decodeRowViaUnpack(b, row, dst)
	case FormatTernaryPacked:
		return decodeRowTernary(b, row, dst)
	case FormatBinaryPacked:
		return decodeRowBinary(b, row, dst)
	default:
		return ErrUnsupported(b.Format, "DecodeRow")
	}
}

func decodeRowViaUnpack(b *Blob, row int, dst []float32) error {
	EnsureFloatCache(b)
	if len(b.F32Cache) >= b.Rows*b.Cols {
		copy(dst[:b.Cols], b.F32Cache[row*b.Cols:(row+1)*b.Cols])
		return nil
	}
	all, err := Unpack(b)
	if err != nil {
		return err
	}
	copy(dst[:b.Cols], all[row*b.Cols:(row+1)*b.Cols])
	return nil
}

func decodeRowQ8_0(b *Blob, row int, dst []float32) error {
	cols := b.Cols
	base := row * cols
	n := b.Rows * cols
	for c := 0; c < cols; c++ {
		i := base + c
		bi := i / q8_0BlockWeights
		j := i % q8_0BlockWeights
		off := bi * q8_0BlockBytes
		if off+4+j >= len(b.Raw) || i >= n {
			dst[c] = 0
			continue
		}
		scale := getF32(b.Raw[off:])
		dst[c] = float32(int8(b.Raw[off+4+j])) * scale
	}
	return nil
}

func decodeRowQ4_0(b *Blob, row int, dst []float32) error {
	cols := b.Cols
	base := row * cols
	n := b.Rows * cols
	for c := 0; c < cols; c++ {
		i := base + c
		if i >= n {
			dst[c] = 0
			continue
		}
		bi := i / q4_0BlockWeights
		j := i % q4_0BlockWeights
		off := bi * q4_0BlockBytes
		if off+4+j/2 >= len(b.Raw) {
			dst[c] = 0
			continue
		}
		scale := getF32(b.Raw[off:])
		w := b.Raw[off+4+j/2]
		var q int8
		if j%2 == 0 {
			q = int8(w & 0xF)
		} else {
			q = int8(w >> 4)
		}
		if q > 7 {
			q -= 16
		}
		dst[c] = float32(q) * scale
	}
	return nil
}

func decodeRowTernary(b *Blob, row int, dst []float32) error {
	cols := b.Cols
	base := row * cols
	n := b.Rows * cols
	for c := 0; c < cols; c++ {
		i := base + c
		if i >= n {
			dst[c] = 0
			continue
		}
		g := i / ternaryGroup
		j := i % ternaryGroup
		if g*4+4 > len(b.Raw) {
			dst[c] = 0
			continue
		}
		word := getU32(b.Raw[g*4:])
		scale := float32(1)
		if g < len(b.Scales) {
			scale = b.Scales[g]
		}
		dst[c] = ternaryCodeToVal(word>>uint(j*2), scale)
	}
	return nil
}

func decodeRowBinary(b *Blob, row int, dst []float32) error {
	if isBinaryG128(b) {
		return decodeRowBinaryG128(b, row, dst)
	}
	cols := b.Cols
	base := row * cols
	n := b.Rows * cols
	for c := 0; c < cols; c++ {
		i := base + c
		if i >= n {
			dst[c] = 0
			continue
		}
		g := i / binaryGroup
		j := i % binaryGroup
		if g*4+4 > len(b.Raw) {
			dst[c] = 0
			continue
		}
		word := getU32(b.Raw[g*4:])
		scale := float32(1)
		if g < len(b.Scales) {
			scale = b.Scales[g]
		}
		if (word>>uint(j))&1 != 0 {
			dst[c] = scale
		} else {
			dst[c] = -scale
		}
	}
	return nil
}
