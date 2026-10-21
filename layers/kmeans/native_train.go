package kmeans

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/quant"
)

func outputModeCAB(m OutputMode) int32 {
	if m == OutputFeatures {
		return 1
	}
	return 0
}

// TrainNativeMSE runs C++ KMeans NormalBP with lr>0. Centers W_after stays pending
// until SyncWeightsFromNative.
func TrainNativeMSE[T core.Numeric](l *Layer, input, target *core.Tensor[T], lr float64) (loss float64, err error) {
	if l == nil || l.Centers == nil || l.Centers.Weights == nil || input == nil || target == nil {
		return 0, fmt.Errorf("kmeans: TrainNativeMSE nil")
	}
	if lr == 0 {
		return 0, fmt.Errorf("kmeans: TrainNativeMSE lr must be > 0")
	}
	if l.Centers.Weights.Format != quant.FormatNone || !nativecpp.NativeCPUAvailable() {
		return 0, fmt.Errorf("kmeans: TrainNativeMSE needs FormatNone + libwelvet")
	}
	lay, err := parseIn(l.Cfg, input)
	if err != nil {
		return 0, err
	}
	w, err := l.Centers.NativeMatrixF32()
	if err != nil {
		return 0, err
	}
	batch, seq := lay.batch, 1
	if len(lay.shape) == 3 {
		batch, seq = lay.shape[0], lay.shape[1]
	}
	outDim := l.Cfg.OutDim()
	nOut := batch * seq * outDim
	nIn := batch * seq * lay.d
	nw := len(w)
	xF := core.SliceAsFloat32(input.Data)
	tF := core.SliceAsFloat32(target.Data)
	preF := make([]float32, nOut)
	postF := make([]float32, nOut)
	dX := make([]float32, nIn)
	dW := make([]float32, nw)
	wAfter := make([]float32, nw)
	loss, err = nativecpp.KMeansNormalBPF32(
		int32(batch), int32(seq), int32(lay.k), int32(lay.d),
		outputModeCAB(l.Cfg.OutputMode), int32(l.Cfg.Activation), l.Cfg.temp(),
		int32(l.Centers.Weights.DType), nativecpp.CABBackendCPU,
		w, xF, tF, true, lr,
		preF, postF, dX, dW, wAfter,
	)
	if err != nil {
		return 0, err
	}
	l.Centers.EnsureNativeMirror().Capture(wAfter, l.Centers.Weights)
	return loss, nil
}

func (l *Layer) SyncWeightsFromNative() error {
	if l == nil || l.Centers == nil {
		return nil
	}
	return l.Centers.SyncWeightsFromNative()
}

func (l *Layer) NativeWeightsPending() bool {
	return l != nil && l.Centers != nil && l.Centers.NativeWeightsPending()
}
