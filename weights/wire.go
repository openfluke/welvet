package weights

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
)

// WireKind selects the host compute representation for a weight store.
// Nothing is hardcoded to float32 — callers pick the wire that matches the dtype.
type WireKind int

const (
	WireF32 WireKind = iota // Plan 9 DotTile / WebGPU f32 SSBO / ggml unpack
	WireF64                 // float64 / wide integer / complex real / high-precision path
	WireI8                  // native int8 + DotI8Tile
	WireU8                  // native uint8 + DotU8Tile
)

// SelectWire chooses the compute wire for FormatNone dtype (or WireF32 for block quants,
// which are defined on f32 ggml layouts — promoted to f64 at MAC time when needed).
func SelectWire(s *Store) WireKind {
	if s == nil {
		return WireF64
	}
	if s.Format != quant.FormatNone {
		return WireF32 // packed blob unpacks to f32 by format definition
	}
	switch s.DType {
	case core.DTypeFloat32, core.DTypeFloat16, core.DTypeBFloat16,
		core.DTypeFP8E4M3, core.DTypeFP8E5M2, core.DTypeFP4, core.DTypeNF4, core.DTypeFP6:
		return WireF32
	case core.DTypeInt8:
		return WireI8
	case core.DTypeUint8:
		return WireU8
	default:
		return WireF64
	}
}

// WireF64 returns a float64 compute-wire view of all weights (cached).
func (s *Store) WireF64() ([]float64, error) {
	if s == nil {
		return nil, fmt.Errorf("weights: nil")
	}
	if s.wireF64 != nil {
		return s.wireF64, nil
	}
	n := s.Rows * s.Cols
	out := make([]float64, n)
	if s.Format != quant.FormatNone {
		f32, err := s.GPUWireF32()
		if err != nil {
			return nil, err
		}
		for i := 0; i < n; i++ {
			out[i] = float64(f32[i])
		}
		s.wireF64 = out
		return out, nil
	}
	row := make([]float64, s.Cols)
	for r := 0; r < s.Rows; r++ {
		if err := DecodeRowF64(s, r, row); err != nil {
			return nil, err
		}
		copy(out[r*s.Cols:], row)
	}
	s.wireF64 = out
	return out, nil
}

// WireRowF64 copies one weight row into dst as float64.
func WireRowF64(s *Store, row int, dst []float64) error {
	if s == nil {
		return fmt.Errorf("weights: nil")
	}
	if row < 0 || row >= s.Rows || len(dst) < s.Cols {
		return fmt.Errorf("weights: WireRowF64 shape")
	}
	if s.Format == quant.FormatNone {
		return DecodeRowF64(s, row, dst)
	}
	w, err := s.WireF64()
	if err != nil {
		return err
	}
	copy(dst[:s.Cols], w[row*s.Cols:(row+1)*s.Cols])
	return nil
}

// DecodeRowF64 streams one FormatNone row into float64 (no f32 bottleneck).
func DecodeRowF64(s *Store, row int, dst []float64) error {
	if s == nil {
		return fmt.Errorf("weights: nil")
	}
	if s.Format != quant.FormatNone {
		return fmt.Errorf("weights: DecodeRowF64 only FormatNone")
	}
	if row < 0 || row >= s.Rows || len(dst) < s.Cols {
		return fmt.Errorf("weights: DecodeRowF64 shape")
	}
	off := row * s.Cols
	n := s.Rows * s.Cols
	if s.DType == core.DTypeFloat32 {
		w, ok := s.MasterF32()
		if !ok {
			return fmt.Errorf("weights: float32 master missing")
		}
		for c := 0; c < s.Cols; c++ {
			dst[c] = float64(w[off+c])
		}
		return nil
	}
	if s.DType == core.DTypeFloat64 {
		raw := s.Native
		if len(raw) < n*8 {
			return fmt.Errorf("weights: float64 native short")
		}
		for c := 0; c < s.Cols; c++ {
			i := off + c
			dst[c] = math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:]))
		}
		return nil
	}
	for c := 0; c < s.Cols; c++ {
		v, err := weightAt(s, off+c, n)
		if err != nil {
			return err
		}
		dst[c] = float64(v)
	}
	return nil
}
