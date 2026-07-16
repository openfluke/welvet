package weights

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
)

// DecodeRow writes one FormatNone weight row into dst (len >= Cols) by streaming
// from native storage — no full-matrix unpack. Used by Dense SIMD DotTile paths.
func DecodeRow(s *Store, row int, dst []float32) error {
	if s == nil {
		return fmt.Errorf("weights: nil")
	}
	if s.Format != quant.FormatNone {
		return fmt.Errorf("weights: DecodeRow only FormatNone (got %s)", s.Format)
	}
	if row < 0 || row >= s.Rows {
		return fmt.Errorf("weights: row %d out of range", row)
	}
	if len(dst) < s.Cols {
		return fmt.Errorf("weights: dst short")
	}
	off := row * s.Cols
	n := s.Rows * s.Cols

	if s.DType == core.DTypeFloat32 {
		w, ok := s.MasterF32()
		if !ok {
			return fmt.Errorf("weights: float32 master missing")
		}
		copy(dst[:s.Cols], w[off:off+s.Cols])
		return nil
	}

	for c := 0; c < s.Cols; c++ {
		v, err := weightAt(s, off+c, n)
		if err != nil {
			return err
		}
		dst[c] = v
	}
	return nil
}

// weightAt streams a single FormatNone weight as float32 for SIMD tile dots.
func weightAt(s *Store, i, n int) (float32, error) {
	scale := s.Scale
	if scale == 0 {
		scale = 1
	}
	raw := s.Native
	dt := s.DType

	switch dt {
	case core.DTypeFloat64:
		if len(raw) < n*8 {
			return 0, fmt.Errorf("weights: float64 short")
		}
		return float32(math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:]))), nil
	case core.DTypeFloat16:
		if len(raw) < n*2 {
			return 0, fmt.Errorf("weights: %s short", dt)
		}
		return core.Float16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:])), nil
	case core.DTypeFP8E4M3:
		if len(raw) < n {
			return 0, fmt.Errorf("weights: fp8e4m3 short")
		}
		return core.FP8E4M3ToFloat32(raw[i]), nil
	case core.DTypeFP8E5M2:
		if len(raw) < n {
			return 0, fmt.Errorf("weights: fp8e5m2 short")
		}
		return core.FP8E5M2ToFloat32(raw[i]), nil
	case core.DTypeFP4:
		if len(raw) < (n+1)/2 {
			return 0, fmt.Errorf("weights: fp4 short")
		}
		var code uint8
		if i%2 == 0 {
			code = raw[i/2] & 0xF
		} else {
			code = raw[i/2] >> 4
		}
		return core.FP4ToFloat32(code), nil
	case core.DTypeBFloat16:
		if len(raw) < n*2 {
			return 0, fmt.Errorf("weights: bfloat16 short")
		}
		return core.BFloat16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:])), nil
	case core.DTypeInt64:
		if len(raw) < n*8 {
			return 0, fmt.Errorf("weights: int64 short")
		}
		return float32(int64(binary.LittleEndian.Uint64(raw[i*8:]))) * scale, nil
	case core.DTypeInt32:
		if len(raw) < n*4 {
			return 0, fmt.Errorf("weights: int32 short")
		}
		return float32(int32(binary.LittleEndian.Uint32(raw[i*4:]))) * scale, nil
	case core.DTypeInt16:
		if len(raw) < n*2 {
			return 0, fmt.Errorf("weights: int16 short")
		}
		return float32(int16(binary.LittleEndian.Uint16(raw[i*2:]))) * scale, nil
	case core.DTypeInt8:
		if len(raw) < n {
			return 0, fmt.Errorf("weights: int8 short")
		}
		return float32(int8(raw[i])) * scale, nil
	case core.DTypeUint8:
		if len(raw) < 4+n {
			return 0, fmt.Errorf("weights: uint8 short")
		}
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		return float32(raw[4+i])*scale + minV, nil
	case core.DTypeUint16:
		if len(raw) < 4+n*2 {
			return 0, fmt.Errorf("weights: uint16 short")
		}
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		return float32(binary.LittleEndian.Uint16(raw[4+i*2:]))*scale + minV, nil
	case core.DTypeUint32:
		if len(raw) < 4+n*4 {
			return 0, fmt.Errorf("weights: uint32 short")
		}
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		return float32(binary.LittleEndian.Uint32(raw[4+i*4:]))*scale + minV, nil
	case core.DTypeUint64:
		if len(raw) < 4+n*8 {
			return 0, fmt.Errorf("weights: uint64 short")
		}
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		return float32(binary.LittleEndian.Uint64(raw[4+i*8:]))*scale + minV, nil
	case core.DTypeInt4:
		if len(raw) < (n+1)/2 {
			return 0, fmt.Errorf("weights: int4 short")
		}
		var q int8
		if i%2 == 0 {
			q = int8(raw[i/2] & 0xF)
		} else {
			q = int8(raw[i/2] >> 4)
		}
		if q > 7 {
			q -= 16
		}
		return float32(q) * scale, nil
	case core.DTypeUint4:
		if len(raw) < 4+(n+1)/2 {
			return 0, fmt.Errorf("weights: uint4 short")
		}
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		body := raw[4:]
		var q int
		if i%2 == 0 {
			q = int(body[i/2] & 0xF)
		} else {
			q = int(body[i/2] >> 4)
		}
		return float32(q)*scale + minV, nil
	case core.DTypeInt2:
		if len(raw) < (n+3)/4 {
			return 0, fmt.Errorf("weights: int2 short")
		}
		code := int((raw[i/4] >> (uint(i%4) * 2)) & 3)
		return float32(code-2) * scale, nil
	case core.DTypeUint2:
		if len(raw) < 4+(n+3)/4 {
			return 0, fmt.Errorf("weights: uint2 short")
		}
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		body := raw[4:]
		code := int((body[i/4] >> (uint(i%4) * 2)) & 3)
		return float32(code)*scale + minV, nil
	case core.DTypeTernary:
		if len(raw) < (n+3)/4 {
			return 0, fmt.Errorf("weights: ternary short")
		}
		code := (raw[i/4] >> (uint(i%4) * 2)) & 3
		switch code {
		case 0:
			return -scale, nil
		case 2:
			return scale, nil
		default:
			return 0, nil
		}
	case core.DTypeBinary:
		if len(raw) < (n+7)/8 {
			return 0, fmt.Errorf("weights: binary short")
		}
		if raw[i/8]&(1<<uint(i%8)) != 0 {
			return scale, nil
		}
		return -scale, nil
	case core.DTypeInt, core.DTypeUint, core.DTypeUintptr,
		core.DTypeComplex64, core.DTypeComplex128,
		core.DTypeNF4, core.DTypeFP6,
		core.DTypeInt6, core.DTypeUint6, core.DTypeInt5, core.DTypeUint5,
		core.DTypeInt3, core.DTypeUint3:
		return decodeExt(dt, raw, scale, i, n)
	default:
		return 0, fmt.Errorf("weights: weightAt unsupported %s", dt)
	}
}
