package quant

import "fmt"

// Format is a native weight packing layout (ggml / llama.cpp k-quants + BitNet).
// Welvet never treats these as QAT overlays — pack is the storage truth.
type Format int

const (
	FormatNone Format = iota

	// Classic ggml blocks
	FormatQ8_0
	FormatQ4_0
	FormatQ4_1
	FormatQ5_0
	FormatQ5_1

	// k-quants (super-block)
	FormatQ2_K
	FormatQ3_K
	FormatQ4_K
	FormatQ5_K
	FormatQ6_K

	// Importance-matrix / IQ family
	FormatIQ1_S
	FormatIQ2_XXS
	FormatIQ2_XS
	FormatIQ3_XXS
	FormatIQ3_S
	FormatIQ4_NL
	FormatIQ4_XS

	// Extreme packs
	FormatTernaryPacked // BitNet 1.58
	FormatBinaryPacked
	FormatAffinePacked // MLX AffineQuantized (e.g. 4-bit g64)
)

// AllFormats is the full quant matrix Welvet must implement natively.
var AllFormats = []Format{
	FormatNone,
	FormatQ8_0, FormatQ4_0, FormatQ4_1, FormatQ5_0, FormatQ5_1,
	FormatQ2_K, FormatQ3_K, FormatQ4_K, FormatQ5_K, FormatQ6_K,
	FormatIQ1_S, FormatIQ2_XXS, FormatIQ2_XS, FormatIQ3_XXS, FormatIQ3_S, FormatIQ4_NL, FormatIQ4_XS,
	FormatTernaryPacked, FormatBinaryPacked, FormatAffinePacked,
}

func (f Format) String() string {
	switch f {
	case FormatNone:
		return "none"
	case FormatQ8_0:
		return "Q8_0"
	case FormatQ4_0:
		return "Q4_0"
	case FormatQ4_1:
		return "Q4_1"
	case FormatQ5_0:
		return "Q5_0"
	case FormatQ5_1:
		return "Q5_1"
	case FormatQ2_K:
		return "Q2_K"
	case FormatQ3_K:
		return "Q3_K"
	case FormatQ4_K:
		return "Q4_K"
	case FormatQ5_K:
		return "Q5_K"
	case FormatQ6_K:
		return "Q6_K"
	case FormatIQ1_S:
		return "IQ1_S"
	case FormatIQ2_XXS:
		return "IQ2_XXS"
	case FormatIQ2_XS:
		return "IQ2_XS"
	case FormatIQ3_XXS:
		return "IQ3_XXS"
	case FormatIQ3_S:
		return "IQ3_S"
	case FormatIQ4_NL:
		return "IQ4_NL"
	case FormatIQ4_XS:
		return "IQ4_XS"
	case FormatTernaryPacked:
		return "TernaryPacked"
	case FormatBinaryPacked:
		return "BinaryPacked"
	case FormatAffinePacked:
		return "AffinePacked"
	default:
		return fmt.Sprintf("Format(%d)", int(f))
	}
}

// IsKQuant reports ggml k-quant super-block formats.
func (f Format) IsKQuant() bool {
	switch f {
	case FormatQ2_K, FormatQ3_K, FormatQ4_K, FormatQ5_K, FormatQ6_K:
		return true
	default:
		return false
	}
}

// Blob is an opaque packed weight buffer + layout metadata.
type Blob struct {
	Format Format
	Rows   int
	Cols   int
	// Raw holds packed bytes (or uint32 words reinterpreted as bytes).
	Raw []byte
	// Scales holds optional host-side scale metadata when not embedded in Raw.
	Scales []float32
	// Mins holds optional per-block minima for asymmetric quants.
	Mins []float32
	// Meta is optional format-specific sidecar data.
	Meta []byte
	// BlockWeights is the primary block / super-block width (e.g. 32 or 256).
	BlockWeights int
	// Q4Packed is optional u32 SIMD view (4 words / Q4_0 block); see EnsureQ4SIMDCache.
	Q4Packed []uint32
	// Int8QS is optional projected int8 codes for Q8_0 / Q5_* / k / IQ / Affine fused SIMD.
	Int8QS []int8
	// F32Cache is optional once-inflated FP32 weights for Ternary/Binary simd_fuse
	// (k/IQ/Affine use Int8QS + Scales/Mins — no full-matrix inflate).
	F32Cache []float32
}

// ParseFormatName maps ENTITY header / blob format strings to Format.
func ParseFormatName(s string) Format {
	if s == "" || s == "none" {
		return FormatNone
	}
	for _, f := range AllFormats {
		if f.String() == s {
			return f
		}
	}
	return FormatNone
}
func Supported(f Format) bool {
	switch f {
	case FormatNone,
		FormatQ8_0, FormatQ4_0, FormatQ4_1, FormatQ5_0, FormatQ5_1,
		FormatQ2_K, FormatQ3_K, FormatQ4_K, FormatQ5_K, FormatQ6_K,
		FormatIQ1_S, FormatIQ2_XXS, FormatIQ2_XS, FormatIQ3_XXS, FormatIQ3_S, FormatIQ4_NL, FormatIQ4_XS,
		FormatTernaryPacked, FormatBinaryPacked, FormatAffinePacked:
		return true
	default:
		return false
	}
}
