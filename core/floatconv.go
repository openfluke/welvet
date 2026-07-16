package core

import "math"

// Float32ToFloat16 packs IEEE754 binary16 (round-to-nearest-even simplified).
func Float32ToFloat16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16((b >> 16) & 0x8000)
	expo := int((b>>23)&0xff) - 127 + 15
	mant := b & 0x7fffff
	if expo <= 0 {
		return sign
	}
	if expo >= 31 {
		return sign | 0x7c00
	}
	return sign | uint16(expo<<10) | uint16(mant>>13)
}

// Float16ToFloat32 expands binary16.
func Float16ToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	expo := (h >> 10) & 0x1f
	mant := uint32(h & 0x3ff)
	if expo == 0 {
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		// denormal → float32 denormal-ish
		return math.Float32frombits(sign | math.Float32bits(float32(mant)/1024/16384))
	}
	if expo == 31 {
		return math.Float32frombits(sign | 0x7f800000 | (mant << 13))
	}
	return math.Float32frombits(sign | uint32(expo-15+127)<<23 | mant<<13)
}

// Float32ToBFloat16 truncates mantissa (round to nearest via +0x7fff bias optional).
func Float32ToBFloat16(f float32) uint16 {
	return uint16(math.Float32bits(f) >> 16)
}

// BFloat16ToFloat32 expands bfloat16.
func BFloat16ToFloat32(h uint16) float32 {
	return math.Float32frombits(uint32(h) << 16)
}
