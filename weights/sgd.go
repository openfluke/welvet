package weights

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
)

// ApplySGD does w ← w − lr·dW in the store's native dtype lanes (FormatNone)
// or via a temporary unpack→update→repack for packed quants.
//
// FormatNone rules:
//   - float32: update the f32 payload in place (that buffer IS storage)
//   - float64: update Native float64 bits with float64 ALU
//   - every other dtype: decode one element → update → re-encode into Native,
//     keeping Scale / unsigned min fixed. Never retains a parallel masterF32.
//
// Packed formats still need a short-lived f32 scratch (block layouts), but
// SetFromF32 drops it afterward — storage truth stays Packed.
func (s *Store) ApplySGD(dW []float64, lr float64) error {
	if s == nil {
		return fmt.Errorf("weights: nil")
	}
	n := s.Rows * s.Cols
	if len(dW) < n {
		return fmt.Errorf("weights: ApplySGD dW len %d < %d", len(dW), n)
	}
	s.gpuF32 = nil
	s.wireF64 = nil

	if s.Format != quant.FormatNone {
		return s.applySGDPacked(dW[:n], lr)
	}
	switch s.DType {
	case core.DTypeFloat32:
		return s.applySGDF32(dW[:n], lr)
	case core.DTypeFloat64:
		return s.applySGDF64(dW[:n], lr)
	default:
		return s.applySGDNative(dW[:n], lr)
	}
}

func (s *Store) applySGDF32(dW []float64, lr float64) error {
	n := s.Rows * s.Cols
	if len(s.masterF32) < n {
		return fmt.Errorf("weights: float32 master missing")
	}
	for i := 0; i < n; i++ {
		s.masterF32[i] -= float32(lr * dW[i])
	}
	return nil
}

func (s *Store) applySGDF64(dW []float64, lr float64) error {
	n := s.Rows * s.Cols
	raw := s.Native
	if len(raw) < n*8 {
		return fmt.Errorf("weights: float64 native short")
	}
	s.masterF32 = nil
	for i := 0; i < n; i++ {
		v := math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:]))
		v -= lr * dW[i]
		binary.LittleEndian.PutUint64(raw[i*8:], math.Float64bits(v))
	}
	return nil
}

func (s *Store) applySGDNative(dW []float64, lr float64) error {
	n := s.Rows * s.Cols
	if len(s.Native) == 0 {
		return fmt.Errorf("weights: ApplySGD no native payload for %s", s.DType)
	}
	s.masterF32 = nil
	// Probe codec before mutating — if setWeightAt can't encode this dtype,
	// fall back to a temporary scratch re-encode (still drops masterF32).
	if n > 0 {
		v0, err := weightAt(s, 0, n)
		if err != nil {
			return err
		}
		if err := setWeightAt(s, 0, n, v0); err != nil {
			return s.applySGDNativeScratch(dW, lr)
		}
	}
	for i := 0; i < n; i++ {
		v, err := weightAt(s, i, n)
		if err != nil {
			return err
		}
		if err := setWeightAt(s, i, n, float32(float64(v)-lr*dW[i])); err != nil {
			return err
		}
	}
	return nil
}

// applySGDNativeScratch updates via a temporary f32 buffer that is never kept
// as masterF32 — only Native (+ Scale) remains after packNative.
func (s *Store) applySGDNativeScratch(dW []float64, lr float64) error {
	n := s.Rows * s.Cols
	scratch, err := unpackNative(s.DType, s.Native, s.Scale, n)
	if err != nil {
		return err
	}
	for i := 0; i < n; i++ {
		scratch[i] -= float32(lr * dW[i])
	}
	raw, scale, err := packNative(s.DType, scratch)
	if err != nil {
		return err
	}
	s.Native = raw
	s.Scale = scale
	s.masterF32 = nil
	s.Packed = nil
	s.Format = quant.FormatNone
	return nil
}

func (s *Store) applySGDPacked(dW []float64, lr float64) error {
	scratch, err := s.float32View()
	if err != nil {
		return err
	}
	// Copy so Unpack's return isn't mutated if shared; then re-Pack.
	tmp := append([]float32(nil), scratch...)
	n := s.Rows * s.Cols
	for i := 0; i < n; i++ {
		tmp[i] -= float32(lr * dW[i])
	}
	return s.SetFromF32(tmp)
}

// ApplyBiasSGD updates Bias ← Bias − lr·dB (float64).
func (s *Store) ApplyBiasSGD(dB []float64, lr float64) error {
	if s == nil {
		return fmt.Errorf("weights: nil")
	}
	if s.Bias == nil {
		return nil
	}
	n := len(s.Bias)
	if len(dB) < n {
		return fmt.Errorf("weights: ApplyBiasSGD dB short")
	}
	for i := 0; i < n; i++ {
		s.Bias[i] -= lr * dB[i]
	}
	return nil
}

// RetainsF32Master reports whether a parallel f32 buffer is kept as storage.
// Only FormatNone+Float32 uses the f32 buffer as payload; every other dtype
// must leave masterF32 empty after SGD / Convert.
func (s *Store) RetainsF32Master() bool {
	if s == nil {
		return false
	}
	return s.Format == quant.FormatNone && s.DType == core.DTypeFloat32 && len(s.masterF32) > 0
}

// F32BufferLen returns len of the internal float32 buffer.
// FormatNone+Float32: this is the payload. Any other dtype/format must be 0
// after ApplySGD / Convert (no retained scratch).
func (s *Store) F32BufferLen() int {
	if s == nil {
		return 0
	}
	return len(s.masterF32)
}
