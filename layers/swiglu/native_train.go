package swiglu

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/quant"
)

// TrainNativeMSE runs C++ SwiGLU NormalBP with lr>0. Gate|Up|Down W_after stays
// pending until SyncWeightsFromNative (for .entity save).
func TrainNativeMSE[T core.Numeric](l *Layer, input, target *core.Tensor[T], lr float64) (loss float64, err error) {
	if l == nil || l.Gate == nil || l.Up == nil || l.Down == nil || input == nil || target == nil {
		return 0, fmt.Errorf("swiglu: TrainNativeMSE nil")
	}
	if lr == 0 {
		return 0, fmt.Errorf("swiglu: TrainNativeMSE lr must be > 0")
	}
	if l.Gate.Weights == nil || l.Gate.Weights.Format != quant.FormatNone || !nativecpp.NativeCPUAvailable() {
		return 0, fmt.Errorf("swiglu: TrainNativeMSE needs FormatNone + libwelvet")
	}
	lay, err := parseLayout(l.Cfg.InputDim, input)
	if err != nil {
		return 0, err
	}
	in, inter := l.Cfg.InputDim, l.Cfg.IntermediateDim
	bs := lay.batch * lay.seq
	gate, err := l.Gate.NativeMatrixF32()
	if err != nil {
		return 0, err
	}
	up, err := l.Up.NativeMatrixF32()
	if err != nil {
		return 0, err
	}
	down, err := l.Down.NativeMatrixF32()
	if err != nil {
		return 0, err
	}
	xF := core.SliceAsFloat32(input.Data)
	tF := core.SliceAsFloat32(target.Data)
	preF := make([]float32, bs*inter)
	postF := make([]float32, bs*in)
	dX := make([]float32, bs*in)
	dW := make([]float32, len(gate)+len(up)+len(down))
	wAfter := make([]float32, len(dW))
	loss, err = nativecpp.SwiGLUNormalBPF32(
		int32(lay.batch), int32(lay.seq), int32(in), int32(inter),
		int32(l.Gate.Weights.DType), nativecpp.CABBackendCPU,
		gate, up, down, xF, tF, true, lr,
		preF, postF, dX, dW, wAfter,
	)
	if err != nil {
		return 0, err
	}
	off := 0
	l.Gate.EnsureNativeMirror().Capture(wAfter[off:off+len(gate)], l.Gate.Weights)
	off += len(gate)
	l.Up.EnsureNativeMirror().Capture(wAfter[off:off+len(up)], l.Up.Weights)
	off += len(up)
	l.Down.EnsureNativeMirror().Capture(wAfter[off:off+len(down)], l.Down.Weights)
	l.ensureMirror().Capture(wAfter, l.Gate.Weights, l.Up.Weights, l.Down.Weights)
	return loss, nil
}

// SyncWeightsFromNative pulls pending C++ Gate/Up/Down into Go for .entity save.
func (l *Layer) SyncWeightsFromNative() error {
	if l == nil {
		return nil
	}
	if l.nativeMir != nil {
		if err := l.nativeMir.SyncToGo(); err != nil {
			return err
		}
	}
	if err := l.Gate.SyncWeightsFromNative(); err != nil {
		return err
	}
	if err := l.Up.SyncWeightsFromNative(); err != nil {
		return err
	}
	return l.Down.SyncWeightsFromNative()
}

// NativeWeightsPending reports whether a SyncWeightsFromNative would write.
func (l *Layer) NativeWeightsPending() bool {
	if l == nil {
		return false
	}
	if l.nativeMir != nil && l.nativeMir.Dirty() {
		return true
	}
	return l.Gate.NativeWeightsPending() || l.Up.NativeWeightsPending() || l.Down.NativeWeightsPending()
}

func (l *Layer) ensureMirror() *nativecpp.Mirror {
	if l.nativeMir == nil {
		l.nativeMir = &nativecpp.Mirror{}
	}
	return l.nativeMir
}
