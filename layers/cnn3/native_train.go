package cnn3

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/quant"
)

// TrainNativeMSE runs C++ CNN3 NormalBP with lr>0. W_after stays pending on Proj
// until SyncWeightsFromNative (for .entity / Go save).
func TrainNativeMSE[T core.Numeric](l *Layer, input, target *core.Tensor[T], lr float64) (loss float64, err error) {
	if l == nil || l.Proj == nil || l.Proj.Weights == nil || input == nil || target == nil {
		return 0, fmt.Errorf("cnn3: TrainNativeMSE nil")
	}
	if lr == 0 {
		return 0, fmt.Errorf("cnn3: TrainNativeMSE lr must be > 0")
	}
	if l.Proj.Weights.Format != quant.FormatNone || !nativecpp.NativeCPUAvailable() {
		return 0, fmt.Errorf("cnn3: TrainNativeMSE needs FormatNone + libwelvet")
	}
	lay, err := parseLayout(l.Cfg, input)
	if err != nil {
		return 0, err
	}
	w, err := l.Proj.NativeMatrixF32()
	if err != nil {
		return 0, err
	}
	sp := lay.outD * lay.outH * lay.outW
	nOut := lay.batch * lay.filters * sp
	nIn := lay.batch * lay.inC * lay.inD * lay.inH * lay.inW
	nw := len(w)
	xF := core.SliceAsFloat32(input.Data)
	tF := core.SliceAsFloat32(target.Data)
	preF := make([]float32, nOut)
	postF := make([]float32, nOut)
	dX := make([]float32, nIn)
	dW := make([]float32, nw)
	wAfter := make([]float32, nw)
	loss, err = nativecpp.CNN3NormalBPF32(
		int32(lay.batch), int32(lay.inC), int32(lay.filters),
		int32(lay.inD), int32(lay.inH), int32(lay.inW),
		int32(l.Cfg.Kernel), int32(l.Cfg.Stride), int32(l.Cfg.Padding),
		int32(l.Cfg.Activation), int32(l.Proj.Weights.DType), nativecpp.CABBackendCPU,
		w, l.Proj.Weights.Bias, xF, tF, true, lr,
		preF, postF, dX, dW, wAfter,
	)
	if err != nil {
		return 0, err
	}
	l.Proj.EnsureNativeMirror().Capture(wAfter, l.Proj.Weights)
	return loss, nil
}

// SyncWeightsFromNative pulls pending C++ Proj weights into Go for .entity save.
func (l *Layer) SyncWeightsFromNative() error {
	if l == nil || l.Proj == nil {
		return nil
	}
	return l.Proj.SyncWeightsFromNative()
}

// NativeWeightsPending reports whether SyncWeightsFromNative would write.
func (l *Layer) NativeWeightsPending() bool {
	return l != nil && l.Proj != nil && l.Proj.NativeWeightsPending()
}
