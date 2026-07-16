package quant

// Unpack expands a Blob to float32 (debug / reference path only).
func Unpack(b *Blob) ([]float32, error) {
	if b == nil {
		return nil, errFormat("Unpack", b)
	}
	switch b.Format {
	case FormatNone:
		return unpackNone(b)
	case FormatQ8_0:
		return unpackQ8_0(b)
	case FormatQ4_0:
		return UnpackQ4_0(b)
	case FormatQ4_1:
		return unpackQ4_1(b)
	case FormatQ5_0:
		return unpackQ5_0(b)
	case FormatQ5_1:
		return unpackQ5_1(b)
	case FormatQ2_K, FormatQ3_K, FormatQ4_K, FormatQ5_K, FormatQ6_K:
		return unpackK(b)
	case FormatIQ1_S, FormatIQ2_XXS, FormatIQ2_XS, FormatIQ3_XXS, FormatIQ3_S, FormatIQ4_NL, FormatIQ4_XS:
		return unpackIQ(b)
	case FormatTernaryPacked:
		return unpackTernary(b)
	case FormatBinaryPacked:
		return unpackBinary(b)
	default:
		return nil, ErrUnsupported(b.Format, "Unpack")
	}
}
