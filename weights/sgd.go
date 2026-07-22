package weights

import (
	"fmt"
)

// materializeMaster decodes the current payload into the FormatNone+F32 buffer
// so SGD can mutate weights before re-encoding storage truth.
func (s *Store) materializeMaster() error {
	if s == nil {
		return fmt.Errorf("weights: nil")
	}
	n := s.Rows * s.Cols
	if len(s.masterF32) >= n {
		return nil
	}
	v, err := s.float32View()
	if err != nil {
		return err
	}
	s.masterF32 = append([]float32(nil), v...)
	return nil
}

// ApplySGD does w ← w − lr·dW on the master source, then re-encodes FormatNone native
// and/or re-Packs the active quant format. Invalidates wire caches.
//
// This is the shared update path for Dense today and other layers that own a Store.
func (s *Store) ApplySGD(dW []float64, lr float64) error {
	if s == nil {
		return fmt.Errorf("weights: nil")
	}
	n := s.Rows * s.Cols
	if len(dW) < n {
		return fmt.Errorf("weights: ApplySGD dW len %d < %d", len(dW), n)
	}
	if err := s.materializeMaster(); err != nil {
		return err
	}
	for i := 0; i < n; i++ {
		s.masterF32[i] -= float32(lr * dW[i])
	}
	s.gpuF32 = nil
	s.wireF64 = nil
	// Re-encode from the updated master. Do NOT call SetDType here: float32View for
	// non-F32 FormatNone reads Native (stale) and would wipe the SGD step.
	return s.SetFromF32(s.masterF32[:n])
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
