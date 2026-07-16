package weights

import (
	"encoding/binary"
	"fmt"
	"math"
	"strconv"

	"github.com/openfluke/welvet/core"
)

// NF4 quantile codes (bitsandbytes / QLoRA), index 0..15 → value in [-1, 1].
var nf4Table = [16]float32{
	-1.0,
	-0.6961928009986877,
	-0.5250730514526367,
	-0.39491748809814453,
	-0.28444138169288635,
	-0.18477343022823334,
	-0.09105003625154495,
	0.0,
	0.07958029955625534,
	0.16093020141124725,
	0.24611230194568634,
	0.33791524171829224,
	0.44070982933044434,
	0.5626170039176941,
	0.7229568362236023,
	1.0,
}

func packExt(dt core.DType, w []float32) ([]byte, float32, error) {
	switch dt {
	case core.DTypeInt:
		return packGoInt(w)
	case core.DTypeUint:
		return packGoUint(w)
	case core.DTypeUintptr:
		return packUintptr(w)
	case core.DTypeComplex64:
		return packComplex64(w)
	case core.DTypeComplex128:
		return packComplex128(w)
	case core.DTypeNF4:
		return packNF4(w)
	case core.DTypeFP6:
		return packFP6(w)
	case core.DTypeInt6:
		return packSignedNBit(w, 6)
	case core.DTypeUint6:
		return packUnsignedNBit(w, 6)
	case core.DTypeInt5:
		return packSignedNBit(w, 5)
	case core.DTypeUint5:
		return packUnsignedNBit(w, 5)
	case core.DTypeInt3:
		return packSignedNBit(w, 3)
	case core.DTypeUint3:
		return packUnsignedNBit(w, 3)
	default:
		return nil, 0, fmt.Errorf("weights: pack ext unsupported %s", dt)
	}
}

func packGoInt(w []float32) ([]byte, float32, error) {
	scale := absMax(w)
	if scale == 0 {
		scale = 1
	}
	n := len(w)
	if strconv.IntSize == 64 {
		b := make([]byte, n*8)
		for i, v := range w {
			q := int64(math.Round(float64(v / scale * 1e6)))
			binary.LittleEndian.PutUint64(b[i*8:], uint64(q))
		}
		return b, scale / 1e6, nil
	}
	b := make([]byte, n*4)
	maxQ := float64(math.MaxInt32)
	for i, v := range w {
		q := int32(clampI(math.Round(float64(v/scale)*maxQ), -maxQ, maxQ))
		binary.LittleEndian.PutUint32(b[i*4:], uint32(q))
	}
	return b, scale / float32(maxQ), nil
}

func packGoUint(w []float32) ([]byte, float32, error) {
	minV, maxV := minMax(w)
	span := maxV - minV
	if span == 0 {
		span = 1
	}
	n := len(w)
	if strconv.IntSize == 64 {
		b := make([]byte, 4+n*8)
		binary.LittleEndian.PutUint32(b, math.Float32bits(minV))
		for i, v := range w {
			q := uint64(clampI(math.Round(float64((v-minV)/span*1e15)), 0, 1e15))
			binary.LittleEndian.PutUint64(b[4+i*8:], q)
		}
		return b, span / 1e15, nil
	}
	b := make([]byte, 4+n*4)
	binary.LittleEndian.PutUint32(b, math.Float32bits(minV))
	maxU := float64(math.MaxUint32)
	for i, v := range w {
		q := uint32(clampI(math.Round(float64((v-minV)/span)*maxU), 0, maxU))
		binary.LittleEndian.PutUint32(b[4+i*4:], q)
	}
	return b, span / float32(maxU), nil
}

func packUintptr(w []float32) ([]byte, float32, error) {
	// Portable wire: always uint64 codes.
	minV, maxV := minMax(w)
	span := maxV - minV
	if span == 0 {
		span = 1
	}
	n := len(w)
	b := make([]byte, 4+n*8)
	binary.LittleEndian.PutUint32(b, math.Float32bits(minV))
	for i, v := range w {
		q := uint64(clampI(math.Round(float64((v-minV)/span*1e15)), 0, 1e15))
		binary.LittleEndian.PutUint64(b[4+i*8:], q)
	}
	return b, span / 1e15, nil
}

func packComplex64(w []float32) ([]byte, float32, error) {
	n := len(w)
	b := make([]byte, n*8) // real + imag (imag=0)
	for i, v := range w {
		binary.LittleEndian.PutUint32(b[i*8:], math.Float32bits(v))
		binary.LittleEndian.PutUint32(b[i*8+4:], 0)
	}
	return b, 1, nil
}

func packComplex128(w []float32) ([]byte, float32, error) {
	n := len(w)
	b := make([]byte, n*16)
	for i, v := range w {
		binary.LittleEndian.PutUint64(b[i*16:], math.Float64bits(float64(v)))
		binary.LittleEndian.PutUint64(b[i*16+8:], 0)
	}
	return b, 1, nil
}

func packNF4(w []float32) ([]byte, float32, error) {
	scale := absMax(w)
	if scale == 0 {
		scale = 1
	}
	n := len(w)
	b := make([]byte, (n+1)/2)
	for i, v := range w {
		r := v / scale
		best, bestErr := 0, float32(math.MaxFloat32)
		for c, t := range nf4Table {
			e := float32(math.Abs(float64(r - t)))
			if e < bestErr {
				bestErr = e
				best = c
			}
		}
		if i%2 == 0 {
			b[i/2] = byte(best & 0xF)
		} else {
			b[i/2] |= byte((best & 0xF) << 4)
		}
	}
	return b, scale, nil
}

func packFP6(w []float32) ([]byte, float32, error) {
	// E3M2-style: 64 levels via scale into signed 6-bit codes [-32..31].
	return packSignedNBit(w, 6)
}

func packSignedNBit(w []float32, bits int) ([]byte, float32, error) {
	if bits < 2 || bits > 7 {
		return nil, 0, fmt.Errorf("weights: bad signed bits %d", bits)
	}
	maxQ := (1 << (bits - 1)) - 1 // e.g. 6-bit → 31
	minQ := -(1 << (bits - 1))    // e.g. 6-bit → -32
	scale := absMax(w) / float32(maxQ)
	if scale == 0 {
		scale = 1
	}
	n := len(w)
	codes := make([]uint64, n)
	for i, v := range w {
		q := int(math.Round(float64(v / scale)))
		if q > maxQ {
			q = maxQ
		}
		if q < minQ {
			q = minQ
		}
		codes[i] = uint64(uint(q) & ((1 << bits) - 1))
	}
	return packBitCodes(codes, bits), scale, nil
}

func packUnsignedNBit(w []float32, bits int) ([]byte, float32, error) {
	if bits < 1 || bits > 7 {
		return nil, 0, fmt.Errorf("weights: bad unsigned bits %d", bits)
	}
	maxQ := (1 << bits) - 1
	minV, maxV := minMax(w)
	span := maxV - minV
	if span == 0 {
		span = 1
	}
	n := len(w)
	codes := make([]uint64, n)
	for i, v := range w {
		q := int(clampI(math.Round(float64((v-minV)/span*float32(maxQ))), 0, float64(maxQ)))
		codes[i] = uint64(q)
	}
	b := packBitCodes(codes, bits)
	out := make([]byte, 4+len(b))
	binary.LittleEndian.PutUint32(out, math.Float32bits(minV))
	copy(out[4:], b)
	return out, span / float32(maxQ), nil
}

func packBitCodes(codes []uint64, bits int) []byte {
	n := len(codes)
	totalBits := n * bits
	b := make([]byte, (totalBits+7)/8)
	for i, c := range codes {
		bitOff := i * bits
		for k := 0; k < bits; k++ {
			if (c>>uint(k))&1 != 0 {
				b[(bitOff+k)/8] |= 1 << uint((bitOff+k)%8)
			}
		}
	}
	return b
}

func getBitCode(raw []byte, i, bits int) uint64 {
	bitOff := i * bits
	var c uint64
	for k := 0; k < bits; k++ {
		if raw[(bitOff+k)/8]&(1<<uint((bitOff+k)%8)) != 0 {
			c |= 1 << uint(k)
		}
	}
	return c
}

func signExtend(code uint64, bits int) int64 {
	signBit := uint64(1) << uint(bits-1)
	mask := (uint64(1) << uint(bits)) - 1
	code &= mask
	if code&signBit != 0 {
		return int64(code) - int64(1<<uint(bits))
	}
	return int64(code)
}

func decodeExt(dt core.DType, raw []byte, scale float32, i, n int) (float32, error) {
	if scale == 0 {
		scale = 1
	}
	switch dt {
	case core.DTypeInt:
		if strconv.IntSize == 64 {
			if len(raw) < n*8 {
				return 0, fmt.Errorf("weights: int native short")
			}
			return float32(int64(binary.LittleEndian.Uint64(raw[i*8:]))) * scale, nil
		}
		if len(raw) < n*4 {
			return 0, fmt.Errorf("weights: int native short")
		}
		return float32(int32(binary.LittleEndian.Uint32(raw[i*4:]))) * scale, nil

	case core.DTypeUint:
		if strconv.IntSize == 64 {
			if len(raw) < 4+n*8 {
				return 0, fmt.Errorf("weights: uint native short")
			}
			minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
			return float32(binary.LittleEndian.Uint64(raw[4+i*8:]))*scale + minV, nil
		}
		if len(raw) < 4+n*4 {
			return 0, fmt.Errorf("weights: uint native short")
		}
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		return float32(binary.LittleEndian.Uint32(raw[4+i*4:]))*scale + minV, nil

	case core.DTypeUintptr:
		if len(raw) < 4+n*8 {
			return 0, fmt.Errorf("weights: uintptr native short")
		}
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		return float32(binary.LittleEndian.Uint64(raw[4+i*8:]))*scale + minV, nil

	case core.DTypeComplex64:
		if len(raw) < n*8 {
			return 0, fmt.Errorf("weights: complex64 native short")
		}
		// GEMV uses real part (imag stored for fidelity).
		return math.Float32frombits(binary.LittleEndian.Uint32(raw[i*8:])), nil

	case core.DTypeComplex128:
		if len(raw) < n*16 {
			return 0, fmt.Errorf("weights: complex128 native short")
		}
		return float32(math.Float64frombits(binary.LittleEndian.Uint64(raw[i*16:]))), nil

	case core.DTypeNF4:
		if len(raw) < (n+1)/2 {
			return 0, fmt.Errorf("weights: nf4 native short")
		}
		var code int
		if i%2 == 0 {
			code = int(raw[i/2] & 0xF)
		} else {
			code = int(raw[i/2] >> 4)
		}
		return nf4Table[code] * scale, nil

	case core.DTypeFP6, core.DTypeInt6:
		need := (n*6 + 7) / 8
		if len(raw) < need {
			return 0, fmt.Errorf("weights: %s native short", dt)
		}
		return float32(signExtend(getBitCode(raw, i, 6), 6)) * scale, nil

	case core.DTypeUint6:
		need := 4 + (n*6+7)/8
		if len(raw) < need {
			return 0, fmt.Errorf("weights: uint6 native short")
		}
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		return float32(getBitCode(raw[4:], i, 6))*scale + minV, nil

	case core.DTypeInt5:
		need := (n*5 + 7) / 8
		if len(raw) < need {
			return 0, fmt.Errorf("weights: int5 native short")
		}
		return float32(signExtend(getBitCode(raw, i, 5), 5)) * scale, nil

	case core.DTypeUint5:
		need := 4 + (n*5+7)/8
		if len(raw) < need {
			return 0, fmt.Errorf("weights: uint5 native short")
		}
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		return float32(getBitCode(raw[4:], i, 5))*scale + minV, nil

	case core.DTypeInt3:
		need := (n*3 + 7) / 8
		if len(raw) < need {
			return 0, fmt.Errorf("weights: int3 native short")
		}
		return float32(signExtend(getBitCode(raw, i, 3), 3)) * scale, nil

	case core.DTypeUint3:
		need := 4 + (n*3+7)/8
		if len(raw) < need {
			return 0, fmt.Errorf("weights: uint3 native short")
		}
		minV := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		return float32(getBitCode(raw[4:], i, 3))*scale + minV, nil

	default:
		return 0, fmt.Errorf("weights: decode ext %s", dt)
	}
}

func unpackExt(dt core.DType, raw []byte, scale float32, n int) ([]float32, error) {
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		v, err := decodeExt(dt, raw, scale, i, n)
		if err != nil {
			return nil, err
		}
		out[i] = v
	}
	return out, nil
}

func matVecViaDecode(s *Store, x, y []float32) error {
	rows, cols := s.Rows, s.Cols
	n := rows * cols
	for r := 0; r < rows; r++ {
		sum := float32(0)
		off := r * cols
		for c := 0; c < cols; c++ {
			w, err := decodeExt(s.DType, s.Native, s.Scale, off+c, n)
			if err != nil {
				return err
			}
			sum += w * x[c]
		}
		y[r] = sum
	}
	return nil
}

func matVecTViaDecode(s *Store, gy, gx []float32) error {
	rows, cols := s.Rows, s.Cols
	n := rows * cols
	for r := 0; r < rows; r++ {
		g := gy[r]
		if g == 0 {
			continue
		}
		off := r * cols
		for c := 0; c < cols; c++ {
			w, err := decodeExt(s.DType, s.Native, s.Scale, off+c, n)
			if err != nil {
				return err
			}
			gx[c] += w * g
		}
	}
	return nil
}
