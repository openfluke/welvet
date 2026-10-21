package parallel

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/nativecpp"
)

// TrainNativeMSE is unavailable: C ABI has Parallel forward only (no NormalBP pack).
// Use Dense branch TrainNativeMSE or Go Backward+ApplyGradSGD instead.
func TrainNativeMSE[T core.Numeric](l *Layer, input, target *core.Tensor[T], lr float64) (loss float64, err error) {
	_ = nativecpp.NativeCPUAvailable
	return 0, fmt.Errorf("parallel: TrainNativeMSE not available (no ParallelNormalBPF32 C ABI)")
}

// SyncWeightsFromNative pulls pending native weights from Dense branches / Gate.
func (l *Layer) SyncWeightsFromNative() error {
	if l == nil {
		return nil
	}
	for i, b := range l.Branches {
		if err := syncBranch(b); err != nil {
			return fmt.Errorf("parallel branch %d: %w", i, err)
		}
	}
	if l.Gate != nil {
		if err := l.Gate.SyncWeightsFromNative(); err != nil {
			return err
		}
	}
	return nil
}

func syncBranch(b any) error {
	switch v := b.(type) {
	case *dense.Layer:
		return v.SyncWeightsFromNative()
	case interface{ SyncWeightsFromNative() error }:
		return v.SyncWeightsFromNative()
	default:
		return nil
	}
}

// NativeWeightsPending reports whether any Dense child / Gate has pending native weights.
func (l *Layer) NativeWeightsPending() bool {
	if l == nil {
		return false
	}
	for _, b := range l.Branches {
		if pendingBranch(b) {
			return true
		}
	}
	return l.Gate != nil && l.Gate.NativeWeightsPending()
}

func pendingBranch(b any) bool {
	switch v := b.(type) {
	case *dense.Layer:
		return v.NativeWeightsPending()
	case interface{ NativeWeightsPending() bool }:
		return v.NativeWeightsPending()
	default:
		return false
	}
}
