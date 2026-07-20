package weights

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
)

// Snapshot is the on-disk / JSON storage truth for one Store (no QAT morph).
// FormatNone×Float32 → LE float32 of master; FormatNone×other → Native+Scale;
// packed formats → EncodePackedWire(Packed).
type Snapshot struct {
	DType  core.DType
	Format quant.Format
	Rows   int
	Cols   int
	Scale  float32
	Raw    []byte
	Bias   []float64
}

// TakeSnapshot captures storage-truth bytes from s.
func TakeSnapshot(s *Store) (Snapshot, error) {
	if s == nil {
		return Snapshot{}, fmt.Errorf("weights: nil store")
	}
	snap := Snapshot{
		DType:  s.DType,
		Format: s.Format,
		Rows:   s.Rows,
		Cols:   s.Cols,
		Scale:  s.Scale,
	}
	if len(s.Bias) > 0 {
		snap.Bias = append([]float64(nil), s.Bias...)
	}
	if s.Format != quant.FormatNone {
		if s.Packed == nil {
			return Snapshot{}, fmt.Errorf("weights: packed format %s missing blob", s.Format)
		}
		snap.Raw = EncodePackedWire(s.Packed)
		return snap, nil
	}
	if s.DType == core.DTypeFloat32 {
		n := s.Rows * s.Cols
		if len(s.masterF32) < n {
			return Snapshot{}, fmt.Errorf("weights: float32 master short")
		}
		snap.Raw = EncodeF32LE(s.masterF32[:n])
		if snap.Scale == 0 {
			snap.Scale = 1
		}
		return snap, nil
	}
	if len(s.Native) == 0 {
		return Snapshot{}, fmt.Errorf("weights: no native payload for %s", s.DType)
	}
	snap.Raw = append([]byte(nil), s.Native...)
	return snap, nil
}

// Restore builds a Store from a storage-truth Snapshot (bit-stable for native/packed).
func Restore(snap Snapshot) (*Store, error) {
	if snap.Rows <= 0 || snap.Cols <= 0 {
		return nil, fmt.Errorf("weights: bad snapshot shape %dx%d", snap.Rows, snap.Cols)
	}
	scale := snap.Scale
	if scale == 0 {
		scale = 1
	}
	if snap.Format != quant.FormatNone {
		b, err := DecodePackedWire(snap.Format, snap.Rows, snap.Cols, snap.Raw)
		if err != nil {
			return nil, err
		}
		s, err := FromBlob(b)
		if err != nil {
			return nil, err
		}
		if len(snap.Bias) > 0 {
			s.Bias = append([]float64(nil), snap.Bias...)
		}
		return s, nil
	}
	s := &Store{
		DType:  snap.DType,
		Format: quant.FormatNone,
		Rows:   snap.Rows,
		Cols:   snap.Cols,
		Scale:  scale,
	}
	if len(snap.Bias) > 0 {
		s.Bias = append([]float64(nil), snap.Bias...)
	}
	n := snap.Rows * snap.Cols
	if snap.DType == core.DTypeFloat32 {
		w, err := DecodeF32LE(snap.Raw, n)
		if err != nil {
			return nil, err
		}
		s.masterF32 = w
		return s, nil
	}
	s.Native = append([]byte(nil), snap.Raw...)
	// Keep master bridge so Pack / Flatten still work after reload.
	flat, err := unpackNative(snap.DType, s.Native, scale, n)
	if err != nil {
		return nil, err
	}
	s.masterF32 = flat
	return s, nil
}

// EncodeF32LE writes little-endian float32 bytes.
func EncodeF32LE(w []float32) []byte {
	buf := make([]byte, len(w)*4)
	for i, v := range w {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

// DecodeF32LE reads LE float32 weights.
func DecodeF32LE(raw []byte, n int) ([]float32, error) {
	if n < 0 {
		return nil, fmt.Errorf("weights: negative count")
	}
	if len(raw) < n*4 {
		return nil, fmt.Errorf("weights: short f32 %d < %d", len(raw), n*4)
	}
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out, nil
}

// EncodeF64LE writes little-endian float64 bytes (bias sidecars).
func EncodeF64LE(w []float64) []byte {
	buf := make([]byte, len(w)*8)
	for i, v := range w {
		binary.LittleEndian.PutUint64(buf[i*8:], math.Float64bits(v))
	}
	return buf
}

// DecodeF64LE reads LE float64 values.
func DecodeF64LE(raw []byte) ([]float64, error) {
	if len(raw)%8 != 0 {
		return nil, fmt.Errorf("weights: f64 length %d not multiple of 8", len(raw))
	}
	n := len(raw) / 8
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:]))
	}
	return out, nil
}
