package core

import "math"

// FP8 E4M3 (OCP E4M3FN-style): 1 sign, 4 exp, 3 mant, bias 7.
// No inf; max finite ≈ 448. Used for DTypeFP8E4M3 native storage (1 byte/elem).

func Float32ToFP8E4M3(f float32) uint8 {
	if f == 0 || math.IsNaN(float64(f)) {
		return 0
	}
	sign := uint8(0)
	if f < 0 {
		sign = 0x80
		f = -f
	}
	if f > 448 {
		return sign | 0x7e // max finite 0b0_1111_110
	}
	bits := math.Float32bits(f)
	exp := int((bits>>23)&0xff) - 127
	mant := bits & 0x7fffff
	e := exp + 7
	if e <= 0 {
		// denormal / underflow → 0 or smallest denorm
		if e < -2 {
			return sign
		}
		mant |= 0x800000
		shift := uint(1 - e + 20) // rough
		m := mant >> shift
		return sign | uint8(m&0x7)
	}
	if e >= 15 {
		return sign | 0x7e
	}
	m := (mant >> 20) & 0x7
	// round
	if mant&0x80000 != 0 {
		m++
		if m == 8 {
			m = 0
			e++
			if e >= 15 {
				return sign | 0x7e
			}
		}
	}
	return sign | uint8(e<<3) | uint8(m)
}

func FP8E4M3ToFloat32(b uint8) float32 {
	sign := uint32(0)
	if b&0x80 != 0 {
		sign = 0x80000000
	}
	e := (b >> 3) & 0xf
	m := b & 0x7
	if e == 0 {
		if m == 0 {
			return math.Float32frombits(sign)
		}
		// denormal
		return math.Float32frombits(sign) * float32(m) / 8 / 64 // 2^(1-7)=1/64
	}
	if e == 15 && m == 7 {
		// FN: max finite, not NaN
		return math.Float32frombits(sign | math.Float32bits(448))
	}
	fexp := uint32(e - 7 + 127)
	return math.Float32frombits(sign | fexp<<23 | uint32(m)<<20)
}

// FP8 E5M2: 1 sign, 5 exp, 2 mant, bias 15 (IEEE-like, has Inf/NaN).

func Float32ToFP8E5M2(f float32) uint8 {
	if math.IsNaN(float64(f)) {
		return 0x7e
	}
	sign := uint8(0)
	if f < 0 || math.Signbit(float64(f)) {
		sign = 0x80
		if f < 0 {
			f = -f
		}
	}
	if math.IsInf(float64(f), 1) {
		return sign | 0x7c
	}
	if f == 0 {
		return sign
	}
	bits := math.Float32bits(f)
	exp := int((bits>>23)&0xff) - 127
	mant := bits & 0x7fffff
	e := exp + 15
	if e <= 0 {
		return sign
	}
	if e >= 31 {
		return sign | 0x7c
	}
	m := (mant >> 21) & 0x3
	if mant&0x100000 != 0 {
		m++
		if m == 4 {
			m = 0
			e++
			if e >= 31 {
				return sign | 0x7c
			}
		}
	}
	return sign | uint8(e<<2) | uint8(m)
}

func FP8E5M2ToFloat32(b uint8) float32 {
	sign := uint32(0)
	if b&0x80 != 0 {
		sign = 0x80000000
	}
	e := (b >> 2) & 0x1f
	m := b & 0x3
	if e == 0 {
		if m == 0 {
			return math.Float32frombits(sign)
		}
		return math.Float32frombits(sign) * float32(m) / 4 / 16384 // 2^(1-15)
	}
	if e == 31 {
		if m == 0 {
			return math.Float32frombits(sign | 0x7f800000)
		}
		return math.Float32frombits(sign | 0x7fc00000)
	}
	fexp := uint32(e - 15 + 127)
	return math.Float32frombits(sign | fexp<<23 | uint32(m)<<21)
}
