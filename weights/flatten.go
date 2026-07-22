package weights

import (
	"fmt"

	"github.com/openfluke/welvet/quant"
)

// FlattenF32 returns a copy of the full weight matrix as float32
// (scale-applied / unpacked). Safe for DNA fingerprints and evolution splice.
func (s *Store) FlattenF32() ([]float32, error) {
	v, err := s.float32View()
	if err != nil {
		return nil, err
	}
	out := make([]float32, len(v))
	copy(out, v)
	return out, nil
}

// SetFromF32 replaces the stored payload with src re-encoded as the current
// DType / QuantFormat. Length must equal Rows*Cols. No parallel f32 copy is kept
// unless the destination is FormatNone+Float32.
func (s *Store) SetFromF32(src []float32) error {
	if s == nil {
		return fmt.Errorf("weights: nil")
	}
	n := s.Rows * s.Cols
	if len(src) < n {
		return fmt.Errorf("weights: SetFromF32 len %d < %d", len(src), n)
	}
	fmtSave := s.Format
	dtSave := s.DType
	if fmtSave == quant.FormatNone {
		return encodeFormatNone(s, dtSave, src[:n])
	}
	return encodePacked(s, fmtSave, src[:n])
}

// Clone returns a deep copy of the store (CPU-resident; no GPU wire caches).
func (s *Store) Clone() (*Store, error) {
	if s == nil {
		return nil, fmt.Errorf("weights: nil")
	}
	flat, err := s.FlattenF32()
	if err != nil {
		return nil, err
	}
	dst, err := New(s.Rows, s.Cols, flat, s.DType, quant.FormatNone)
	if err != nil {
		return nil, err
	}
	if len(s.Bias) > 0 {
		dst.Bias = append([]float64(nil), s.Bias...)
	}
	if s.Format != quant.FormatNone {
		if err := dst.Pack(s.Format); err != nil {
			return nil, err
		}
	}
	return dst, nil
}

// ParamCount returns Rows*Cols (+ bias length when present).
func (s *Store) ParamCount() int {
	if s == nil {
		return 0
	}
	n := s.Rows * s.Cols
	if s.Bias != nil {
		n += len(s.Bias)
	}
	return n
}
