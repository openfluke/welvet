package weights

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/openfluke/welvet/core"
)

func packNative(dt core.DType, w []float32) ([]byte, float32, error) {
	n := len(w)
	switch dt {
	case core.DTypeFloat64:
		b := make([]byte, n*8)
		for i, v := range w {
			binary.LittleEndian.PutUint64(b[i*8:], math.Float64bits(float64(v)))
		}
		return b, 1, nil
	case core.DTypeFloat16:
		b := make([]byte, n*2)
		for i, v := range w {
			binary.LittleEndian.PutUint16(b[i*2:], core.Float32ToFloat16(v))
		}
		return b, 1, nil
	case core.DTypeBFloat16:
		b := make([]byte, n*2)
		for i, v := range w {
			binary.LittleEndian.PutUint16(b[i*2:], core.Float32ToBFloat16(v))
		}
		return b, 1, nil
	case core.DTypeFP8E4M3:
		b := make([]byte, n)
		for i, v := range w {
			b[i] = core.Float32ToFP8E4M3(v)
		}
		return b, 1, nil
	case core.DTypeFP8E5M2:
		b := make([]byte, n)
		for i, v := range w {
			b[i] = core.Float32ToFP8E5M2(v)
		}
		return b, 1, nil
	case core.DTypeFP4:
		b := make([]byte, (n+1)/2)
		for i := 0; i < n; i++ {
			code := core.Float32ToFP4(w[i]) & 0xF
			if i%2 == 0 {
				b[i/2] = code
			} else {
				b[i/2] |= code << 4
			}
		}
		return b, 1, nil
	case core.DTypeInt64:
		b := make([]byte, n*8)
		scale := absMax(w)
		if scale == 0 {
			scale = 1
		}
		for i, v := range w {
			q := int64(math.Round(float64(v / scale * 1e6))) // retain relative range
			binary.LittleEndian.PutUint64(b[i*8:], uint64(q))
		}
		return b, scale / 1e6, nil
	case core.DTypeInt32:
		b := make([]byte, n*4)
		scale := absMax(w)
		if scale == 0 {
			scale = 1
		}
		for i, v := range w {
			q := int32(math.Round(float64(v / scale * 2147483647)))
			binary.LittleEndian.PutUint32(b[i*4:], uint32(q))
		}
		return b, scale / 2147483647, nil
	case core.DTypeInt16:
		b := make([]byte, n*2)
		scale := absMax(w)
		if scale == 0 {
			scale = 1
		}
		for i, v := range w {
			q := int16(clampI(math.Round(float64(v/scale*32767)), -32768, 32767))
			binary.LittleEndian.PutUint16(b[i*2:], uint16(q))
		}
		return b, scale / 32767, nil
	case core.DTypeInt8:
		b := make([]byte, n)
		scale := absMax(w) / 127
		if scale == 0 {
			scale = 1
		}
		for i, v := range w {
			b[i] = byte(int8(clampI(math.Round(float64(v/scale)), -128, 127)))
		}
		return b, scale, nil
	case core.DTypeUint64, core.DTypeUint32, core.DTypeUint16, core.DTypeUint8:
		return packUnsigned(dt, w)
	case core.DTypeInt4:
		return packInt4(w)
	case core.DTypeUint4:
		return packUint4(w)
	case core.DTypeInt2:
		return packInt2(w)
	case core.DTypeUint2:
		return packUint2(w)
	case core.DTypeTernary:
		return packTernary(w)
	case core.DTypeBinary:
		return packBinary(w)
	case core.DTypeInt, core.DTypeUint, core.DTypeUintptr,
		core.DTypeComplex64, core.DTypeComplex128,
		core.DTypeNF4, core.DTypeFP6,
		core.DTypeInt6, core.DTypeUint6, core.DTypeInt5, core.DTypeUint5,
		core.DTypeInt3, core.DTypeUint3:
		return packExt(dt, w)
	default:
		return nil, 0, fmt.Errorf("weights: pack native unsupported %s", dt)
	}
}

func packUnsigned(dt core.DType, w []float32) ([]byte, float32, error) {
	n := len(w)
	minV, maxV := w[0], w[0]
	for _, v := range w[1:] {
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
	}
	span := maxV - minV
	if span == 0 {
		span = 1
	}
	switch dt {
	case core.DTypeUint8:
		b := make([]byte, n)
		for i, v := range w {
			b[i] = byte(clampI(math.Round(float64((v-minV)/span*255)), 0, 255))
		}
		// decode: q/255*span + min → store scale=span/255, bias in Scale high? use Scale=span/255 and encode min into first 4 bytes meta — keep simple: scale maps 0..255 → 0..span, add min via Scale sign hack.
		// Store min in Float32[0] reserved? Cleaner: scale = span/255, and we store min as negative Scale unused.
		// Use Scale for step; Mins not available — bake min into Scale by storing step and keeping min in Native prefix.
		out := make([]byte, 4+n)
		binary.LittleEndian.PutUint32(out, math.Float32bits(minV))
		copy(out[4:], b)
		return out, span / 255, nil
	case core.DTypeUint16:
		b := make([]byte, 4+n*2)
		binary.LittleEndian.PutUint32(b, math.Float32bits(minV))
		for i, v := range w {
			q := uint16(clampI(math.Round(float64((v-minV)/span*65535)), 0, 65535))
			binary.LittleEndian.PutUint16(b[4+i*2:], q)
		}
		return b, span / 65535, nil
	case core.DTypeUint32:
		b := make([]byte, 4+n*4)
		binary.LittleEndian.PutUint32(b, math.Float32bits(minV))
		for i, v := range w {
			q := uint32(clampI(math.Round(float64((v-minV)/span*float32(math.MaxUint32))), 0, float64(math.MaxUint32)))
			binary.LittleEndian.PutUint32(b[4+i*4:], q)
		}
		return b, span / float32(math.MaxUint32), nil
	case core.DTypeUint64:
		b := make([]byte, 4+n*8)
		binary.LittleEndian.PutUint32(b, math.Float32bits(minV))
		for i, v := range w {
			q := uint64(clampI(math.Round(float64((v-minV)/span*1e15)), 0, 1e15))
			binary.LittleEndian.PutUint64(b[4+i*8:], q)
		}
		return b, span / 1e15, nil
	default:
		return nil, 0, fmt.Errorf("weights: unsigned %s", dt)
	}
}

func packInt4(w []float32) ([]byte, float32, error) {
	scale := absMax(w) / 7
	if scale == 0 {
		scale = 1
	}
	n := len(w)
	b := make([]byte, (n+1)/2)
	for i := 0; i < n; i++ {
		q := int(clampI(math.Round(float64(w[i]/scale)), -8, 7)) & 0xF
		if i%2 == 0 {
			b[i/2] = byte(q)
		} else {
			b[i/2] |= byte(q << 4)
		}
	}
	return b, scale, nil
}

func packUint4(w []float32) ([]byte, float32, error) {
	minV, maxV := minMax(w)
	span := maxV - minV
	if span == 0 {
		span = 1
	}
	n := len(w)
	b := make([]byte, 4+(n+1)/2)
	binary.LittleEndian.PutUint32(b, math.Float32bits(minV))
	for i := 0; i < n; i++ {
		q := int(clampI(math.Round(float64((w[i]-minV)/span*15)), 0, 15)) & 0xF
		if i%2 == 0 {
			b[4+i/2] = byte(q)
		} else {
			b[4+i/2] |= byte(q << 4)
		}
	}
	return b, span / 15, nil
}

func packInt2(w []float32) ([]byte, float32, error) {
	scale := absMax(w) / 1
	if scale == 0 {
		scale = 1
	}
	n := len(w)
	b := make([]byte, (n+3)/4)
	for i := 0; i < n; i++ {
		q := int(clampI(math.Round(float64(w[i]/scale)), -2, 1)) // {-2,-1,0,1} → store +2 → {0,1,2,3}
		code := (q + 2) & 3
		b[i/4] |= byte(code << (uint(i%4) * 2))
	}
	return b, scale, nil
}

func packUint2(w []float32) ([]byte, float32, error) {
	minV, maxV := minMax(w)
	span := maxV - minV
	if span == 0 {
		span = 1
	}
	n := len(w)
	b := make([]byte, 4+(n+3)/4)
	binary.LittleEndian.PutUint32(b, math.Float32bits(minV))
	for i := 0; i < n; i++ {
		q := int(clampI(math.Round(float64((w[i]-minV)/span*3)), 0, 3)) & 3
		b[4+i/4] |= byte(q << (uint(i%4) * 2))
	}
	return b, span / 3, nil
}

func packTernary(w []float32) ([]byte, float32, error) {
	scale := absMax(w)
	if scale == 0 {
		scale = 1
	}
	n := len(w)
	b := make([]byte, (n+3)/4)
	for i, v := range w {
		var code int
		r := v / scale
		if r > 0.5 {
			code = 2
		} else if r < -0.5 {
			code = 0
		} else {
			code = 1
		}
		b[i/4] |= byte(code << (uint(i%4) * 2))
	}
	return b, scale, nil
}

func packBinary(w []float32) ([]byte, float32, error) {
	scale := absMax(w)
	if scale == 0 {
		scale = 1
	}
	n := len(w)
	b := make([]byte, (n+7)/8)
	for i, v := range w {
		if v >= 0 {
			b[i/8] |= 1 << uint(i%8)
		}
	}
	return b, scale, nil
}

func unpackNative(dt core.DType, raw []byte, scale float32, n int) ([]float32, error) {
	out := make([]float32, n)
	if scale == 0 {
		scale = 1
	}
	switch dt {
	case core.DTypeFloat64:
		for i := 0; i < n; i++ {
			out[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:])))
		}
	case core.DTypeFloat16:
		for i := 0; i < n; i++ {
			out[i] = core.Float16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
	case core.DTypeFP8E4M3:
		if len(raw) < n {
			return nil, fmt.Errorf("weights: fp8e4m3 short")
		}
		for i := 0; i < n; i++ {
			out[i] = core.FP8E4M3ToFloat32(raw[i])
		}
	case core.DTypeFP8E5M2:
		if len(raw) < n {
			return nil, fmt.Errorf("weights: fp8e5m2 short")
		}
		for i := 0; i < n; i++ {
			out[i] = core.FP8E5M2ToFloat32(raw[i])
		}
	case core.DTypeFP4:
		if len(raw) < (n+1)/2 {
			return nil, fmt.Errorf("weights: fp4 short")
		}
		for i := 0; i < n; i++ {
			var code uint8
			if i%2 == 0 {
				code = raw[i/2] & 0xF
			} else {
				code = raw[i/2] >> 4
			}
			out[i] = core.FP4ToFloat32(code)
		}
	case core.DTypeBFloat16:
		for i := 0; i < n; i++ {
			out[i] = core.BFloat16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
	case core.DTypeInt64:
		for i := 0; i < n; i++ {
			out[i] = float32(int64(binary.LittleEndian.Uint64(raw[i*8:]))) * scale
		}
	case core.DTypeInt32:
		for i := 0; i < n; i++ {
			out[i] = float32(int32(binary.LittleEndian.Uint32(raw[i*4:]))) * scale
		}
	case core.DTypeInt16:
		for i := 0; i < n; i++ {
			out[i] = float32(int16(binary.LittleEndian.Uint16(raw[i*2:]))) * scale
		}
	case core.DTypeInt8:
		for i := 0; i < n; i++ {
			out[i] = float32(int8(raw[i])) * scale
		}
	case core.DTypeUint8:
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		for i := 0; i < n; i++ {
			out[i] = float32(raw[4+i])*scale + minV
		}
	case core.DTypeUint16:
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		for i := 0; i < n; i++ {
			out[i] = float32(binary.LittleEndian.Uint16(raw[4+i*2:]))*scale + minV
		}
	case core.DTypeUint32:
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		for i := 0; i < n; i++ {
			out[i] = float32(binary.LittleEndian.Uint32(raw[4+i*4:]))*scale + minV
		}
	case core.DTypeUint64:
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		for i := 0; i < n; i++ {
			out[i] = float32(binary.LittleEndian.Uint64(raw[4+i*8:]))*scale + minV
		}
	case core.DTypeInt4:
		for i := 0; i < n; i++ {
			var q int8
			if i%2 == 0 {
				q = int8(raw[i/2] & 0xF)
			} else {
				q = int8(raw[i/2] >> 4)
			}
			if q > 7 {
				q -= 16
			}
			out[i] = float32(q) * scale
		}
	case core.DTypeUint4:
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		for i := 0; i < n; i++ {
			var q int
			if i%2 == 0 {
				q = int(raw[4+i/2] & 0xF)
			} else {
				q = int(raw[4+i/2] >> 4)
			}
			out[i] = float32(q)*scale + minV
		}
	case core.DTypeInt2:
		for i := 0; i < n; i++ {
			code := int((raw[i/4] >> (uint(i%4) * 2)) & 3)
			out[i] = float32(code-2) * scale
		}
	case core.DTypeUint2:
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		for i := 0; i < n; i++ {
			code := int((raw[4+i/4] >> (uint(i%4) * 2)) & 3)
			out[i] = float32(code)*scale + minV
		}
	case core.DTypeTernary:
		for i := 0; i < n; i++ {
			code := (raw[i/4] >> (uint(i%4) * 2)) & 3
			var t float32
			switch code {
			case 0:
				t = -1
			case 2:
				t = 1
			default:
				t = 0
			}
			out[i] = t * scale
		}
	case core.DTypeBinary:
		for i := 0; i < n; i++ {
			if raw[i/8]&(1<<uint(i%8)) != 0 {
				out[i] = scale
			} else {
				out[i] = -scale
			}
		}
	case core.DTypeInt, core.DTypeUint, core.DTypeUintptr,
		core.DTypeComplex64, core.DTypeComplex128,
		core.DTypeNF4, core.DTypeFP6,
		core.DTypeInt6, core.DTypeUint6, core.DTypeInt5, core.DTypeUint5,
		core.DTypeInt3, core.DTypeUint3:
		return unpackExt(dt, raw, scale, n)
	default:
		return nil, fmt.Errorf("weights: unpack %s", dt)
	}
	return out, nil
}

// matVecNative runs FormatNone GEMV by streaming weights from native storage.
// It must NOT unpack the full matrix to float32 first — that is the forbidden soft path.
func matVecNative(s *Store, x, y []float32) error {
	rows, cols := s.Rows, s.Cols
	if len(x) < cols || len(y) < rows {
		return fmt.Errorf("weights: matvec shape")
	}
	n := rows * cols
	scale := s.Scale
	if scale == 0 {
		scale = 1
	}
	raw := s.Native

	switch s.DType {
	case core.DTypeFloat32:
		w := s.masterF32
		if len(w) < n {
			return fmt.Errorf("weights: float32 master missing")
		}
		for r := 0; r < rows; r++ {
			sum := float32(0)
			off := r * cols
			for c := 0; c < cols; c++ {
				sum += w[off+c] * x[c]
			}
			y[r] = sum
		}
		return nil

	case core.DTypeFloat64:
		if len(raw) < n*8 {
			return fmt.Errorf("weights: float64 native short")
		}
		for r := 0; r < rows; r++ {
			sum := 0.0
			off := r * cols
			for c := 0; c < cols; c++ {
				sum += math.Float64frombits(binary.LittleEndian.Uint64(raw[(off+c)*8:])) * float64(x[c])
			}
			y[r] = float32(sum)
		}
		return nil

	case core.DTypeFloat16:
		if len(raw) < n*2 {
			return fmt.Errorf("weights: float16 native short")
		}
		for r := 0; r < rows; r++ {
			sum := float32(0)
			off := r * cols
			for c := 0; c < cols; c++ {
				sum += core.Float16ToFloat32(binary.LittleEndian.Uint16(raw[(off+c)*2:])) * x[c]
			}
			y[r] = sum
		}
		return nil

	case core.DTypeFP8E4M3:
		if len(raw) < n {
			return fmt.Errorf("weights: fp8e4m3 native short")
		}
		for r := 0; r < rows; r++ {
			sum := float32(0)
			off := r * cols
			for c := 0; c < cols; c++ {
				sum += core.FP8E4M3ToFloat32(raw[off+c]) * x[c]
			}
			y[r] = sum
		}
		return nil

	case core.DTypeFP8E5M2:
		if len(raw) < n {
			return fmt.Errorf("weights: fp8e5m2 native short")
		}
		for r := 0; r < rows; r++ {
			sum := float32(0)
			off := r * cols
			for c := 0; c < cols; c++ {
				sum += core.FP8E5M2ToFloat32(raw[off+c]) * x[c]
			}
			y[r] = sum
		}
		return nil

	case core.DTypeFP4:
		if len(raw) < (n+1)/2 {
			return fmt.Errorf("weights: fp4 native short")
		}
		for r := 0; r < rows; r++ {
			sum := float32(0)
			off := r * cols
			for c := 0; c < cols; c++ {
				i := off + c
				var code uint8
				if i%2 == 0 {
					code = raw[i/2] & 0xF
				} else {
					code = raw[i/2] >> 4
				}
				sum += core.FP4ToFloat32(code) * x[c]
			}
			y[r] = sum
		}
		return nil

	case core.DTypeBFloat16:
		if len(raw) < n*2 {
			return fmt.Errorf("weights: bfloat16 native short")
		}
		for r := 0; r < rows; r++ {
			sum := float32(0)
			off := r * cols
			for c := 0; c < cols; c++ {
				sum += core.BFloat16ToFloat32(binary.LittleEndian.Uint16(raw[(off+c)*2:])) * x[c]
			}
			y[r] = sum
		}
		return nil

	case core.DTypeInt64:
		if len(raw) < n*8 {
			return fmt.Errorf("weights: int64 native short")
		}
		for r := 0; r < rows; r++ {
			sum := float32(0)
			off := r * cols
			for c := 0; c < cols; c++ {
				sum += float32(int64(binary.LittleEndian.Uint64(raw[(off+c)*8:]))) * scale * x[c]
			}
			y[r] = sum
		}
		return nil

	case core.DTypeInt32:
		if len(raw) < n*4 {
			return fmt.Errorf("weights: int32 native short")
		}
		for r := 0; r < rows; r++ {
			sum := float32(0)
			off := r * cols
			for c := 0; c < cols; c++ {
				sum += float32(int32(binary.LittleEndian.Uint32(raw[(off+c)*4:]))) * scale * x[c]
			}
			y[r] = sum
		}
		return nil

	case core.DTypeInt16:
		if len(raw) < n*2 {
			return fmt.Errorf("weights: int16 native short")
		}
		for r := 0; r < rows; r++ {
			sum := float32(0)
			off := r * cols
			for c := 0; c < cols; c++ {
				sum += float32(int16(binary.LittleEndian.Uint16(raw[(off+c)*2:]))) * scale * x[c]
			}
			y[r] = sum
		}
		return nil

	case core.DTypeInt8:
		if len(raw) < n {
			return fmt.Errorf("weights: int8 native short")
		}
		for r := 0; r < rows; r++ {
			sum := float32(0)
			off := r * cols
			for c := 0; c < cols; c++ {
				sum += float32(int8(raw[off+c])) * scale * x[c]
			}
			y[r] = sum
		}
		return nil

	case core.DTypeUint8:
		if len(raw) < 4+n {
			return fmt.Errorf("weights: uint8 native short")
		}
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		body := raw[4:]
		for r := 0; r < rows; r++ {
			sum := float32(0)
			off := r * cols
			for c := 0; c < cols; c++ {
				sum += (float32(body[off+c])*scale + minV) * x[c]
			}
			y[r] = sum
		}
		return nil

	case core.DTypeUint16:
		if len(raw) < 4+n*2 {
			return fmt.Errorf("weights: uint16 native short")
		}
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		for r := 0; r < rows; r++ {
			sum := float32(0)
			off := r * cols
			for c := 0; c < cols; c++ {
				q := float32(binary.LittleEndian.Uint16(raw[4+(off+c)*2:]))
				sum += (q*scale + minV) * x[c]
			}
			y[r] = sum
		}
		return nil

	case core.DTypeUint32:
		if len(raw) < 4+n*4 {
			return fmt.Errorf("weights: uint32 native short")
		}
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		for r := 0; r < rows; r++ {
			sum := float32(0)
			off := r * cols
			for c := 0; c < cols; c++ {
				q := float32(binary.LittleEndian.Uint32(raw[4+(off+c)*4:]))
				sum += (q*scale + minV) * x[c]
			}
			y[r] = sum
		}
		return nil

	case core.DTypeUint64:
		if len(raw) < 4+n*8 {
			return fmt.Errorf("weights: uint64 native short")
		}
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		for r := 0; r < rows; r++ {
			sum := float32(0)
			off := r * cols
			for c := 0; c < cols; c++ {
				q := float32(binary.LittleEndian.Uint64(raw[4+(off+c)*8:]))
				sum += (q*scale + minV) * x[c]
			}
			y[r] = sum
		}
		return nil

	case core.DTypeInt4:
		if len(raw) < (n+1)/2 {
			return fmt.Errorf("weights: int4 native short")
		}
		for r := 0; r < rows; r++ {
			sum := float32(0)
			off := r * cols
			for c := 0; c < cols; c++ {
				i := off + c
				var q int8
				if i%2 == 0 {
					q = int8(raw[i/2] & 0xF)
				} else {
					q = int8(raw[i/2] >> 4)
				}
				if q > 7 {
					q -= 16
				}
				sum += float32(q) * scale * x[c]
			}
			y[r] = sum
		}
		return nil

	case core.DTypeUint4:
		if len(raw) < 4+(n+1)/2 {
			return fmt.Errorf("weights: uint4 native short")
		}
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		body := raw[4:]
		for r := 0; r < rows; r++ {
			sum := float32(0)
			off := r * cols
			for c := 0; c < cols; c++ {
				i := off + c
				var q int
				if i%2 == 0 {
					q = int(body[i/2] & 0xF)
				} else {
					q = int(body[i/2] >> 4)
				}
				sum += (float32(q)*scale + minV) * x[c]
			}
			y[r] = sum
		}
		return nil

	case core.DTypeInt2:
		if len(raw) < (n+3)/4 {
			return fmt.Errorf("weights: int2 native short")
		}
		for r := 0; r < rows; r++ {
			sum := float32(0)
			off := r * cols
			for c := 0; c < cols; c++ {
				i := off + c
				code := int((raw[i/4] >> (uint(i%4) * 2)) & 3)
				sum += float32(code-2) * scale * x[c]
			}
			y[r] = sum
		}
		return nil

	case core.DTypeUint2:
		if len(raw) < 4+(n+3)/4 {
			return fmt.Errorf("weights: uint2 native short")
		}
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		body := raw[4:]
		for r := 0; r < rows; r++ {
			sum := float32(0)
			off := r * cols
			for c := 0; c < cols; c++ {
				i := off + c
				code := int((body[i/4] >> (uint(i%4) * 2)) & 3)
				sum += (float32(code)*scale + minV) * x[c]
			}
			y[r] = sum
		}
		return nil

	case core.DTypeTernary:
		if len(raw) < (n+3)/4 {
			return fmt.Errorf("weights: ternary native short")
		}
		for r := 0; r < rows; r++ {
			sum := float32(0)
			off := r * cols
			for c := 0; c < cols; c++ {
				i := off + c
				code := (raw[i/4] >> (uint(i%4) * 2)) & 3
				var t float32
				switch code {
				case 0:
					t = -1
				case 2:
					t = 1
				default:
					t = 0
				}
				sum += t * scale * x[c]
			}
			y[r] = sum
		}
		return nil

	case core.DTypeBinary:
		if len(raw) < (n+7)/8 {
			return fmt.Errorf("weights: binary native short")
		}
		for r := 0; r < rows; r++ {
			sum := float32(0)
			off := r * cols
			for c := 0; c < cols; c++ {
				i := off + c
				if raw[i/8]&(1<<uint(i%8)) != 0 {
					sum += scale * x[c]
				} else {
					sum += -scale * x[c]
				}
			}
			y[r] = sum
		}
		return nil

	case core.DTypeInt, core.DTypeUint, core.DTypeUintptr,
		core.DTypeComplex64, core.DTypeComplex128,
		core.DTypeNF4, core.DTypeFP6,
		core.DTypeInt6, core.DTypeUint6, core.DTypeInt5, core.DTypeUint5,
		core.DTypeInt3, core.DTypeUint3:
		return matVecViaDecode(s, x, y)

	default:
		return fmt.Errorf("weights: matVec native unsupported %s", s.DType)
	}
}

func matVecTNative(s *Store, gy, gx []float32) error {
	rows, cols := s.Rows, s.Cols
	if len(gy) < rows || len(gx) < cols {
		return fmt.Errorf("weights: matvecT shape")
	}
	n := rows * cols
	scale := s.Scale
	if scale == 0 {
		scale = 1
	}
	raw := s.Native

	// Shared pattern: for each row r, gx += w[r,:] * gy[r], decoding w from native storage.
	addRow := func(r int, decode func(i int) float32) {
		g := gy[r]
		if g == 0 {
			return
		}
		off := r * cols
		for c := 0; c < cols; c++ {
			gx[c] += decode(off+c) * g
		}
	}

	switch s.DType {
	case core.DTypeFloat32:
		w := s.masterF32
		if len(w) < n {
			return fmt.Errorf("weights: float32 master missing")
		}
		for r := 0; r < rows; r++ {
			g := gy[r]
			off := r * cols
			for c := 0; c < cols; c++ {
				gx[c] += w[off+c] * g
			}
		}
		return nil

	case core.DTypeFloat64:
		if len(raw) < n*8 {
			return fmt.Errorf("weights: float64 native short")
		}
		for r := 0; r < rows; r++ {
			addRow(r, func(i int) float32 {
				return float32(math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:])))
			})
		}
		return nil

	case core.DTypeFloat16:
		if len(raw) < n*2 {
			return fmt.Errorf("weights: float16 native short")
		}
		for r := 0; r < rows; r++ {
			addRow(r, func(i int) float32 {
				return core.Float16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
			})
		}
		return nil

	case core.DTypeFP8E4M3:
		if len(raw) < n {
			return fmt.Errorf("weights: fp8e4m3 native short")
		}
		for r := 0; r < rows; r++ {
			addRow(r, func(i int) float32 { return core.FP8E4M3ToFloat32(raw[i]) })
		}
		return nil

	case core.DTypeFP8E5M2:
		if len(raw) < n {
			return fmt.Errorf("weights: fp8e5m2 native short")
		}
		for r := 0; r < rows; r++ {
			addRow(r, func(i int) float32 { return core.FP8E5M2ToFloat32(raw[i]) })
		}
		return nil

	case core.DTypeFP4:
		if len(raw) < (n+1)/2 {
			return fmt.Errorf("weights: fp4 native short")
		}
		for r := 0; r < rows; r++ {
			addRow(r, func(i int) float32 {
				var code uint8
				if i%2 == 0 {
					code = raw[i/2] & 0xF
				} else {
					code = raw[i/2] >> 4
				}
				return core.FP4ToFloat32(code)
			})
		}
		return nil

	case core.DTypeBFloat16:
		if len(raw) < n*2 {
			return fmt.Errorf("weights: bfloat16 native short")
		}
		for r := 0; r < rows; r++ {
			addRow(r, func(i int) float32 {
				return core.BFloat16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
			})
		}
		return nil

	case core.DTypeInt64:
		if len(raw) < n*8 {
			return fmt.Errorf("weights: int64 native short")
		}
		for r := 0; r < rows; r++ {
			addRow(r, func(i int) float32 {
				return float32(int64(binary.LittleEndian.Uint64(raw[i*8:]))) * scale
			})
		}
		return nil

	case core.DTypeInt32:
		if len(raw) < n*4 {
			return fmt.Errorf("weights: int32 native short")
		}
		for r := 0; r < rows; r++ {
			addRow(r, func(i int) float32 {
				return float32(int32(binary.LittleEndian.Uint32(raw[i*4:]))) * scale
			})
		}
		return nil

	case core.DTypeInt16:
		if len(raw) < n*2 {
			return fmt.Errorf("weights: int16 native short")
		}
		for r := 0; r < rows; r++ {
			addRow(r, func(i int) float32 {
				return float32(int16(binary.LittleEndian.Uint16(raw[i*2:]))) * scale
			})
		}
		return nil

	case core.DTypeInt8:
		if len(raw) < n {
			return fmt.Errorf("weights: int8 native short")
		}
		for r := 0; r < rows; r++ {
			addRow(r, func(i int) float32 { return float32(int8(raw[i])) * scale })
		}
		return nil

	case core.DTypeUint8:
		if len(raw) < 4+n {
			return fmt.Errorf("weights: uint8 native short")
		}
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		body := raw[4:]
		for r := 0; r < rows; r++ {
			addRow(r, func(i int) float32 { return float32(body[i])*scale + minV })
		}
		return nil

	case core.DTypeUint16:
		if len(raw) < 4+n*2 {
			return fmt.Errorf("weights: uint16 native short")
		}
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		for r := 0; r < rows; r++ {
			addRow(r, func(i int) float32 {
				return float32(binary.LittleEndian.Uint16(raw[4+i*2:]))*scale + minV
			})
		}
		return nil

	case core.DTypeUint32:
		if len(raw) < 4+n*4 {
			return fmt.Errorf("weights: uint32 native short")
		}
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		for r := 0; r < rows; r++ {
			addRow(r, func(i int) float32 {
				return float32(binary.LittleEndian.Uint32(raw[4+i*4:]))*scale + minV
			})
		}
		return nil

	case core.DTypeUint64:
		if len(raw) < 4+n*8 {
			return fmt.Errorf("weights: uint64 native short")
		}
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		for r := 0; r < rows; r++ {
			addRow(r, func(i int) float32 {
				return float32(binary.LittleEndian.Uint64(raw[4+i*8:]))*scale + minV
			})
		}
		return nil

	case core.DTypeInt4:
		if len(raw) < (n+1)/2 {
			return fmt.Errorf("weights: int4 native short")
		}
		for r := 0; r < rows; r++ {
			addRow(r, func(i int) float32 {
				var q int8
				if i%2 == 0 {
					q = int8(raw[i/2] & 0xF)
				} else {
					q = int8(raw[i/2] >> 4)
				}
				if q > 7 {
					q -= 16
				}
				return float32(q) * scale
			})
		}
		return nil

	case core.DTypeUint4:
		if len(raw) < 4+(n+1)/2 {
			return fmt.Errorf("weights: uint4 native short")
		}
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		body := raw[4:]
		for r := 0; r < rows; r++ {
			addRow(r, func(i int) float32 {
				var q int
				if i%2 == 0 {
					q = int(body[i/2] & 0xF)
				} else {
					q = int(body[i/2] >> 4)
				}
				return float32(q)*scale + minV
			})
		}
		return nil

	case core.DTypeInt2:
		if len(raw) < (n+3)/4 {
			return fmt.Errorf("weights: int2 native short")
		}
		for r := 0; r < rows; r++ {
			addRow(r, func(i int) float32 {
				code := int((raw[i/4] >> (uint(i%4) * 2)) & 3)
				return float32(code-2) * scale
			})
		}
		return nil

	case core.DTypeUint2:
		if len(raw) < 4+(n+3)/4 {
			return fmt.Errorf("weights: uint2 native short")
		}
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		body := raw[4:]
		for r := 0; r < rows; r++ {
			addRow(r, func(i int) float32 {
				code := int((body[i/4] >> (uint(i%4) * 2)) & 3)
				return float32(code)*scale + minV
			})
		}
		return nil

	case core.DTypeTernary:
		if len(raw) < (n+3)/4 {
			return fmt.Errorf("weights: ternary native short")
		}
		for r := 0; r < rows; r++ {
			addRow(r, func(i int) float32 {
				code := (raw[i/4] >> (uint(i%4) * 2)) & 3
				switch code {
				case 0:
					return -scale
				case 2:
					return scale
				default:
					return 0
				}
			})
		}
		return nil

	case core.DTypeBinary:
		if len(raw) < (n+7)/8 {
			return fmt.Errorf("weights: binary native short")
		}
		for r := 0; r < rows; r++ {
			addRow(r, func(i int) float32 {
				if raw[i/8]&(1<<uint(i%8)) != 0 {
					return scale
				}
				return -scale
			})
		}
		return nil

	case core.DTypeInt, core.DTypeUint, core.DTypeUintptr,
		core.DTypeComplex64, core.DTypeComplex128,
		core.DTypeNF4, core.DTypeFP6,
		core.DTypeInt6, core.DTypeUint6, core.DTypeInt5, core.DTypeUint5,
		core.DTypeInt3, core.DTypeUint3:
		return matVecTViaDecode(s, gy, gx)

	default:
		return fmt.Errorf("weights: matVecT native unsupported %s", s.DType)
	}
}

func matVecNativeF64(s *Store, x, y []float64) error {
	rows, cols := s.Rows, s.Cols
	if len(x) < cols || len(y) < rows {
		return fmt.Errorf("weights: matvecF64 shape")
	}
	wRow := make([]float64, cols)
	for r := 0; r < rows; r++ {
		if err := DecodeRowF64(s, r, wRow); err != nil {
			return err
		}
		sum := 0.0
		for c := 0; c < cols; c++ {
			sum += wRow[c] * x[c]
		}
		y[r] = sum
	}
	return nil
}

func matVecTNativeF64(s *Store, gy, gx []float64) error {
	rows, cols := s.Rows, s.Cols
	if len(gy) < rows || len(gx) < cols {
		return fmt.Errorf("weights: matvecTF64 shape")
	}
	wRow := make([]float64, cols)
	for r := 0; r < rows; r++ {
		g := gy[r]
		if g == 0 {
			continue
		}
		if err := DecodeRowF64(s, r, wRow); err != nil {
			return err
		}
		for c := 0; c < cols; c++ {
			gx[c] += wRow[c] * g
		}
	}
	return nil
}

func absMax(w []float32) float32 {
	m := float32(0)
	for _, v := range w {
		a := float32(math.Abs(float64(v)))
		if a > m {
			m = a
		}
	}
	return m
}

func minMax(w []float32) (float32, float32) {
	minV, maxV := w[0], w[0]
	for _, v := range w[1:] {
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
	}
	return minV, maxV
}

func clampI(v float64, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
