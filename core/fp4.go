package core

import "math"

// FP4 E2M1: 1 sign, 2 exp, 1 mant, bias 1. Sixteen codes; used for DTypeFP4 (nibble pack).

var fp4E2M1Table = [16]float32{
	0, 0.5, 1, 1.5, 2, 3, 4, 6,
	-0, -0.5, -1, -1.5, -2, -3, -4, -6,
}

// Float32ToFP4 returns a 4-bit E2M1 code (0..15).
func Float32ToFP4(f float32) uint8 {
	best := uint8(0)
	bestErr := float32(math.MaxFloat32)
	for c := 0; c < 16; c++ {
		e := float32(math.Abs(float64(f - fp4E2M1Table[c])))
		if e < bestErr {
			bestErr = e
			best = uint8(c)
		}
	}
	return best
}

// FP4ToFloat32 expands a 4-bit E2M1 code.
func FP4ToFloat32(code uint8) float32 {
	return fp4E2M1Table[code&0xf]
}
