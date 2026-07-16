package simd

import (
	"encoding/binary"
	"math"

	"github.com/openfluke/welvet/core"
)

// DotF16Packed — x[f32] · packed IEEE Float16 weights (LE uint16), f64 accum.
// On amd64 uses F16C VCVTPH2PS + AVX2 FMA when linked; otherwise Go convert + DotTile tiles.
func DotF16Packed(x []float32, w []byte, i0, n int, prev float64) float64 {
	if n <= 0 {
		return prev
	}
	if len(w) < (i0+n)*2 {
		return prev
	}
	if simdEnabled() {
		return dotF16PackedSimd(x, w, i0, n, prev)
	}
	return dotF16PackedGo(x, w, i0, n, prev)
}

func dotF16PackedGo(x []float32, w []byte, i0, n int, prev float64) float64 {
	return dotF16PackedTiled(x, w, i0, n, prev)
}

func dotF16PackedTiled(x []float32, w []byte, i0, n int, prev float64) float64 {
	sum := prev
	const tile = 8
	var buf [tile]float32
	for i := 0; i < n; {
		m := tile
		if n-i < m {
			m = n - i
		}
		for k := 0; k < m; k++ {
			off := (i0 + i + k) * 2
			h := binary.LittleEndian.Uint16(w[off:])
			buf[k] = math.Float32frombits(f16bitsToF32bits(h))
		}
		sum = DotTile(x[i:i+m], buf[:m], 0, m, sum)
		i += m
	}
	return sum
}

// DotBF16Packed — x[f32] · packed bfloat16 (high 16 bits of f32).
func DotBF16Packed(x []float32, w []byte, i0, n int, prev float64) float64 {
	if n <= 0 {
		return prev
	}
	if len(w) < (i0+n)*2 {
		return prev
	}
	sum := prev
	const tile = 8
	var buf [tile]float32
	for i := 0; i < n; {
		m := tile
		if n-i < m {
			m = n - i
		}
		for k := 0; k < m; k++ {
			off := (i0 + i + k) * 2
			h := binary.LittleEndian.Uint16(w[off:])
			buf[k] = math.Float32frombits(uint32(h) << 16)
		}
		sum = DotTile(x[i:i+m], buf[:m], 0, m, sum)
		i += m
	}
	return sum
}

// DotFP8Packed — x[f32] · packed FP8 (kind 0=E4M3, 1=E5M2).
func DotFP8Packed(x []float32, w []byte, i0, n, kind int, prev float64) float64 {
	if n <= 0 {
		return prev
	}
	if len(w) < i0+n {
		return prev
	}
	if simdEnabled() {
		return dotFP8PackedSimd(x, w, i0, n, kind, prev)
	}
	return dotFP8PackedGo(x, w, i0, n, kind, prev)
}

func dotFP8PackedGo(x []float32, w []byte, i0, n, kind int, prev float64) float64 {
	sum := prev
	const tile = 8
	var buf [tile]float32
	for i := 0; i < n; {
		m := tile
		if n-i < m {
			m = n - i
		}
		for k := 0; k < m; k++ {
			b := w[i0+i+k]
			if kind == 1 {
				buf[k] = core.FP8E5M2ToFloat32(b)
			} else {
				buf[k] = core.FP8E4M3ToFloat32(b)
			}
		}
		sum = DotTile(x[i:i+m], buf[:m], 0, m, sum)
		i += m
	}
	return sum
}

// DotFP4Packed — x[f32] · nibble-packed E2M1 FP4.
func DotFP4Packed(x []float32, w []byte, i0, n int, prev float64) float64 {
	if n <= 0 {
		return prev
	}
	sum := prev
	const tile = 8
	var buf [tile]float32
	for i := 0; i < n; {
		m := tile
		if n-i < m {
			m = n - i
		}
		for k := 0; k < m; k++ {
			flat := i0 + i + k
			b := w[flat/2]
			var code uint8
			if flat%2 == 0 {
				code = b & 0xf
			} else {
				code = b >> 4
			}
			buf[k] = core.FP4ToFloat32(code)
		}
		sum = DotTile(x[i:i+m], buf[:m], 0, m, sum)
		i += m
	}
	return sum
}

func f16bitsToF32bits(h uint16) uint32 {
	sign := uint32(h&0x8000) << 16
	exp := (h >> 10) & 0x1f
	mant := uint32(h & 0x3ff)
	switch exp {
	case 0:
		if mant == 0 {
			return sign
		}
		// denorm
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		mant &= 0x3ff
		exp++
		return sign | uint32(127-15+int(exp))<<23 | mant<<13
	case 31:
		return sign | 0x7f800000 | mant<<13
	default:
		return sign | uint32(int(exp)+127-15)<<23 | mant<<13
	}
}
