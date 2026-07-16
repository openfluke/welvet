package quant

// Pack converts a row-major float32 matrix into a native Format blob.
func Pack(format Format, weights []float32, rows, cols int) (*Blob, error) {
	switch format {
	case FormatNone:
		return packNone(weights, rows, cols)
	case FormatQ8_0:
		return PackQ8_0(weights, rows, cols)
	case FormatQ4_0:
		return PackQ4_0(weights, rows, cols)
	case FormatQ4_1:
		return packQ4_1(weights, rows, cols)
	case FormatQ5_0:
		return packQ5_0(weights, rows, cols)
	case FormatQ5_1:
		return packQ5_1(weights, rows, cols)
	case FormatQ2_K, FormatQ3_K, FormatQ4_K, FormatQ5_K, FormatQ6_K:
		return packK(format, weights, rows, cols)
	case FormatIQ1_S, FormatIQ2_XXS, FormatIQ2_XS, FormatIQ3_XXS, FormatIQ3_S, FormatIQ4_NL, FormatIQ4_XS:
		return packIQ(format, weights, rows, cols)
	case FormatTernaryPacked:
		return packTernary(weights, rows, cols)
	case FormatBinaryPacked:
		return packBinary(weights, rows, cols)
	default:
		return nil, ErrUnsupported(format, "Pack")
	}
}
