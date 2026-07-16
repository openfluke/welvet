package quant

// MatVec computes y = W @ x for packed W (rows×cols), decoding blocks on the fly.
// y must have length ≥ Rows; x must have length ≥ Cols. y is overwritten.
func MatVec(b *Blob, x, y []float32) error {
	if err := checkBlobYX("MatVec", b, x, y); err != nil {
		return err
	}
	switch b.Format {
	case FormatNone:
		return matVecNone(b, x, y)
	case FormatQ8_0:
		return matVecQ8_0(b, x, y)
	case FormatQ4_0:
		return matVecQ4_0(b, x, y)
	case FormatQ4_1:
		return matVecQ4_1(b, x, y)
	case FormatQ5_0:
		return matVecQ5_0(b, x, y)
	case FormatQ5_1:
		return matVecQ5_1(b, x, y)
	case FormatQ2_K, FormatQ3_K, FormatQ4_K, FormatQ5_K, FormatQ6_K:
		return matVecK(b, x, y)
	case FormatIQ1_S, FormatIQ2_XXS, FormatIQ2_XS, FormatIQ3_XXS, FormatIQ3_S, FormatIQ4_NL, FormatIQ4_XS:
		return matVecIQ(b, x, y)
	case FormatTernaryPacked:
		return matVecTernary(b, x, y)
	case FormatBinaryPacked:
		return matVecBinary(b, x, y)
	default:
		return ErrUnsupported(b.Format, "MatVec")
	}
}
