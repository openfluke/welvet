package metacognition

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/quant"
)

// TrainNativeMSE runs C++ Metacognition NormalBP with lr>0. Observed W_after stays
// pending until SyncWeightsFromNative.
func TrainNativeMSE[T core.Numeric](l *Layer, input, target *core.Tensor[T], lr float64) (loss float64, err error) {
	if l == nil || l.Observed == nil || l.Observed.Weights == nil || input == nil || target == nil {
		return 0, fmt.Errorf("metacognition: TrainNativeMSE nil")
	}
	if lr == 0 {
		return 0, fmt.Errorf("metacognition: TrainNativeMSE lr must be > 0")
	}
	if l.Observed.Weights.Format != quant.FormatNone || !nativecpp.NativeCPUAvailable() {
		return 0, fmt.Errorf("metacognition: TrainNativeMSE needs FormatNone + libwelvet")
	}
	flat, err := flatten(l.Cfg, input)
	if err != nil {
		return 0, err
	}
	dim := l.Cfg.Dim
	batch, seq := flat.Shape[0], 1
	if len(input.Shape) == 3 {
		batch, seq = input.Shape[0], input.Shape[1]
	}
	w, err := l.Observed.NativeMatrixF32()
	if err != nil {
		return 0, err
	}
	n := batch * seq * dim
	nw := len(w)
	xF := core.SliceAsFloat32(input.Data)
	tF := core.SliceAsFloat32(target.Data)
	preF := make([]float32, n)
	postF := make([]float32, n)
	dX := make([]float32, n)
	dW := make([]float32, nw)
	wAfter := make([]float32, nw)
	loss, err = nativecpp.MetacognitionNormalBPF32(
		int32(batch), int32(seq), int32(dim),
		int32(l.Observed.Weights.DType), nativecpp.CABBackendCPU,
		w, xF, tF, true, lr,
		preF, postF, dX, dW, wAfter,
	)
	if err != nil {
		return 0, err
	}
	l.Observed.EnsureNativeMirror().Capture(wAfter, l.Observed.Weights)
	return loss, nil
}

func (l *Layer) SyncWeightsFromNative() error {
	if l == nil || l.Observed == nil {
		return nil
	}
	return l.Observed.SyncWeightsFromNative()
}

func (l *Layer) NativeWeightsPending() bool {
	return l != nil && l.Observed != nil && l.Observed.NativeWeightsPending()
}
