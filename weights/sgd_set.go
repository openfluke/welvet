package weights

import (
	"encoding/binary"
	"fmt"
	"math"
	"strconv"

	"github.com/openfluke/welvet/core"
)

// setWeightAt writes a float32 value into FormatNone Native at index i,
// keeping Scale and unsigned min headers fixed. Mirrors weightAt encoding.
func setWeightAt(s *Store, i, n int, v float32) error {
	scale := s.Scale
	if scale == 0 {
		scale = 1
	}
	raw := s.Native
	dt := s.DType

	switch dt {
	case core.DTypeFloat16:
		if len(raw) < n*2 {
			return fmt.Errorf("weights: float16 short")
		}
		binary.LittleEndian.PutUint16(raw[i*2:], core.Float32ToFloat16(v))
		return nil
	case core.DTypeBFloat16:
		if len(raw) < n*2 {
			return fmt.Errorf("weights: bfloat16 short")
		}
		binary.LittleEndian.PutUint16(raw[i*2:], core.Float32ToBFloat16(v))
		return nil
	case core.DTypeFP8E4M3:
		if len(raw) < n {
			return fmt.Errorf("weights: fp8e4m3 short")
		}
		raw[i] = core.Float32ToFP8E4M3(v)
		return nil
	case core.DTypeFP8E5M2:
		if len(raw) < n {
			return fmt.Errorf("weights: fp8e5m2 short")
		}
		raw[i] = core.Float32ToFP8E5M2(v)
		return nil
	case core.DTypeFP4:
		if len(raw) < (n+1)/2 {
			return fmt.Errorf("weights: fp4 short")
		}
		putNibble(raw, i, core.Float32ToFP4(v))
		return nil
	case core.DTypeInt64:
		if len(raw) < n*8 {
			return fmt.Errorf("weights: int64 short")
		}
		q := int64(math.Round(float64(v / scale)))
		binary.LittleEndian.PutUint64(raw[i*8:], uint64(q))
		return nil
	case core.DTypeInt32:
		if len(raw) < n*4 {
			return fmt.Errorf("weights: int32 short")
		}
		q := int32(clampI(math.Round(float64(v/scale)), math.MinInt32, math.MaxInt32))
		binary.LittleEndian.PutUint32(raw[i*4:], uint32(q))
		return nil
	case core.DTypeInt16:
		if len(raw) < n*2 {
			return fmt.Errorf("weights: int16 short")
		}
		q := int16(clampI(math.Round(float64(v/scale)), -32768, 32767))
		binary.LittleEndian.PutUint16(raw[i*2:], uint16(q))
		return nil
	case core.DTypeInt8:
		if len(raw) < n {
			return fmt.Errorf("weights: int8 short")
		}
		q := int8(clampI(math.Round(float64(v/scale)), -128, 127))
		raw[i] = byte(q)
		return nil
	case core.DTypeUint8:
		if len(raw) < 4+n {
			return fmt.Errorf("weights: uint8 short")
		}
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		q := uint8(clampI(math.Round(float64((v-minV)/scale)), 0, 255))
		raw[4+i] = q
		return nil
	case core.DTypeUint16:
		if len(raw) < 4+n*2 {
			return fmt.Errorf("weights: uint16 short")
		}
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		q := uint16(clampI(math.Round(float64((v-minV)/scale)), 0, 65535))
		binary.LittleEndian.PutUint16(raw[4+i*2:], q)
		return nil
	case core.DTypeUint32:
		if len(raw) < 4+n*4 {
			return fmt.Errorf("weights: uint32 short")
		}
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		q := uint32(clampI(math.Round(float64((v-minV)/scale)), 0, float64(math.MaxUint32)))
		binary.LittleEndian.PutUint32(raw[4+i*4:], q)
		return nil
	case core.DTypeUint64:
		if len(raw) < 4+n*8 {
			return fmt.Errorf("weights: uint64 short")
		}
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		q := uint64(clampI(math.Round(float64((v-minV)/scale)), 0, 1e15))
		binary.LittleEndian.PutUint64(raw[4+i*8:], q)
		return nil
	case core.DTypeInt4:
		if len(raw) < (n+1)/2 {
			return fmt.Errorf("weights: int4 short")
		}
		q := int(clampI(math.Round(float64(v/scale)), -8, 7)) & 0xF
		putNibble(raw, i, uint8(q))
		return nil
	case core.DTypeUint4:
		if len(raw) < 4+(n+1)/2 {
			return fmt.Errorf("weights: uint4 short")
		}
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		q := int(clampI(math.Round(float64((v-minV)/scale)), 0, 15)) & 0xF
		putNibble(raw[4:], i, uint8(q))
		return nil
	case core.DTypeInt2:
		if len(raw) < (n+3)/4 {
			return fmt.Errorf("weights: int2 short")
		}
		q := int(clampI(math.Round(float64(v/scale)), -2, 1))
		put2Bit(raw, i, uint8((q+2)&3))
		return nil
	case core.DTypeUint2:
		if len(raw) < 4+(n+3)/4 {
			return fmt.Errorf("weights: uint2 short")
		}
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		q := int(clampI(math.Round(float64((v-minV)/scale)), 0, 3)) & 3
		put2Bit(raw[4:], i, uint8(q))
		return nil
	case core.DTypeTernary:
		if len(raw) < (n+3)/4 {
			return fmt.Errorf("weights: ternary short")
		}
		r := v / scale
		var code uint8
		if r > 0.5 {
			code = 2
		} else if r < -0.5 {
			code = 0
		} else {
			code = 1
		}
		put2Bit(raw, i, code)
		return nil
	case core.DTypeBinary:
		if len(raw) < (n+7)/8 {
			return fmt.Errorf("weights: binary short")
		}
		bit := uint(i % 8)
		if v >= 0 {
			raw[i/8] |= 1 << bit
		} else {
			raw[i/8] &^= 1 << bit
		}
		return nil
	case core.DTypeInt, core.DTypeUint, core.DTypeUintptr,
		core.DTypeComplex64, core.DTypeComplex128,
		core.DTypeNF4, core.DTypeFP6,
		core.DTypeInt6, core.DTypeUint6, core.DTypeInt5, core.DTypeUint5,
		core.DTypeInt3, core.DTypeUint3:
		return setWeightExt(dt, raw, scale, i, n, v)
	default:
		return fmt.Errorf("weights: setWeightAt unsupported %s", dt)
	}
}

func setWeightExt(dt core.DType, raw []byte, scale float32, i, n int, v float32) error {
	if scale == 0 {
		scale = 1
	}
	switch dt {
	case core.DTypeInt:
		if strconv.IntSize == 64 {
			if len(raw) < n*8 {
				return fmt.Errorf("weights: int short")
			}
			q := int64(math.Round(float64(v / scale)))
			binary.LittleEndian.PutUint64(raw[i*8:], uint64(q))
			return nil
		}
		if len(raw) < n*4 {
			return fmt.Errorf("weights: int short")
		}
		q := int32(clampI(math.Round(float64(v/scale)), math.MinInt32, math.MaxInt32))
		binary.LittleEndian.PutUint32(raw[i*4:], uint32(q))
		return nil
	case core.DTypeUint:
		if strconv.IntSize == 64 {
			if len(raw) < 4+n*8 {
				return fmt.Errorf("weights: uint short")
			}
			minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
			q := uint64(clampI(math.Round(float64((v-minV)/scale)), 0, 1e15))
			binary.LittleEndian.PutUint64(raw[4+i*8:], q)
			return nil
		}
		if len(raw) < 4+n*4 {
			return fmt.Errorf("weights: uint short")
		}
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		q := uint32(clampI(math.Round(float64((v-minV)/scale)), 0, float64(math.MaxUint32)))
		binary.LittleEndian.PutUint32(raw[4+i*4:], q)
		return nil
	case core.DTypeUintptr:
		if len(raw) < 4+n*8 {
			return fmt.Errorf("weights: uintptr short")
		}
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		q := uint64(clampI(math.Round(float64((v-minV)/scale)), 0, 1e15))
		binary.LittleEndian.PutUint64(raw[4+i*8:], q)
		return nil
	case core.DTypeComplex64:
		if len(raw) < n*8 {
			return fmt.Errorf("weights: complex64 short")
		}
		binary.LittleEndian.PutUint32(raw[i*8:], math.Float32bits(v))
		// leave imag as-is
		return nil
	case core.DTypeComplex128:
		if len(raw) < n*16 {
			return fmt.Errorf("weights: complex128 short")
		}
		binary.LittleEndian.PutUint64(raw[i*16:], math.Float64bits(float64(v)))
		return nil
	case core.DTypeNF4:
		if len(raw) < (n+1)/2 {
			return fmt.Errorf("weights: nf4 short")
		}
		r := v / scale
		best, bestErr := 0, float32(math.MaxFloat32)
		for c, t := range nf4Table {
			e := float32(math.Abs(float64(r - t)))
			if e < bestErr {
				bestErr = e
				best = c
			}
		}
		putNibble(raw, i, uint8(best&0xF))
		return nil
	case core.DTypeFP6, core.DTypeInt6:
		return setSignedNBit(raw, i, n, 6, scale, v)
	case core.DTypeInt5:
		return setSignedNBit(raw, i, n, 5, scale, v)
	case core.DTypeInt3:
		return setSignedNBit(raw, i, n, 3, scale, v)
	case core.DTypeUint6:
		return setUnsignedNBit(raw, i, n, 6, scale, v)
	case core.DTypeUint5:
		return setUnsignedNBit(raw, i, n, 5, scale, v)
	case core.DTypeUint3:
		return setUnsignedNBit(raw, i, n, 3, scale, v)
	default:
		return fmt.Errorf("weights: setWeightExt unsupported %s", dt)
	}
}

func setSignedNBit(raw []byte, i, n, bits int, scale, v float32) error {
	maxQ := (1 << (bits - 1)) - 1
	minQ := -(1 << (bits - 1))
	need := (n*bits + 7) / 8
	if len(raw) < need {
		return fmt.Errorf("weights: signed%d short", bits)
	}
	q := int(math.Round(float64(v / scale)))
	if q > maxQ {
		q = maxQ
	}
	if q < minQ {
		q = minQ
	}
	setBitCode(raw, i, bits, uint64(uint(q)&((1<<bits)-1)))
	return nil
}

func setUnsignedNBit(raw []byte, i, n, bits int, scale, v float32) error {
	maxQ := (1 << bits) - 1
	bodyNeed := (n*bits + 7) / 8
	if len(raw) < 4+bodyNeed {
		return fmt.Errorf("weights: unsigned%d short", bits)
	}
	minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
	q := int(clampI(math.Round(float64((v-minV)/scale)), 0, float64(maxQ)))
	setBitCode(raw[4:], i, bits, uint64(q))
	return nil
}

func putNibble(raw []byte, i int, code uint8) {
	code &= 0xF
	if i%2 == 0 {
		raw[i/2] = (raw[i/2] & 0xF0) | code
	} else {
		raw[i/2] = (raw[i/2] & 0x0F) | (code << 4)
	}
}

func put2Bit(raw []byte, i int, code uint8) {
	code &= 3
	shift := uint(i%4) * 2
	raw[i/4] = (raw[i/4] &^ (3 << shift)) | (code << shift)
}

func setBitCode(raw []byte, i, bits int, code uint64) {
	bitOff := i * bits
	for k := 0; k < bits; k++ {
		byteIdx := (bitOff + k) / 8
		bit := uint((bitOff + k) % 8)
		if (code>>uint(k))&1 != 0 {
			raw[byteIdx] |= 1 << bit
		} else {
			raw[byteIdx] &^= 1 << bit
		}
	}
}
