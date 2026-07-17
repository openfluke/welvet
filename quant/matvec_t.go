package quant

// MatVecT accumulates gx += Wᵀ @ gy for packed W (rows×cols), decoding on the fly.
// gy length ≥ Rows; gx length ≥ Cols. gx is not zeroed — caller accumulates.
func MatVecT(b *Blob, gy, gx []float32) error {
	if err := checkBlobGYGX("MatVecT", b, gy, gx); err != nil {
		return err
	}
	switch b.Format {
	case FormatNone:
		return matVecTNone(b, gy, gx)
	case FormatQ8_0:
		return matVecTQ8_0(b, gy, gx)
	case FormatQ4_0:
		return matVecTQ4_0(b, gy, gx)
	case FormatQ4_1:
		return matVecTQ4_1(b, gy, gx)
	case FormatQ5_0:
		return matVecTQ5_0(b, gy, gx)
	case FormatQ5_1:
		return matVecTQ5_1(b, gy, gx)
	case FormatQ2_K, FormatQ3_K, FormatQ4_K, FormatQ5_K, FormatQ6_K:
		return matVecTK(b, gy, gx)
	case FormatIQ1_S, FormatIQ2_XXS, FormatIQ2_XS, FormatIQ3_XXS, FormatIQ3_S, FormatIQ4_NL, FormatIQ4_XS:
		return matVecTIQ(b, gy, gx)
	case FormatTernaryPacked:
		return matVecTTernary(b, gy, gx)
	case FormatBinaryPacked:
		return matVecTBinary(b, gy, gx)
	case FormatAffinePacked:
		return matVecTAffine(b, gy, gx)
	default:
		return ErrUnsupported(b.Format, "MatVecT")
	}
}
