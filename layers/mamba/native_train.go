package mamba

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/quant"
)

// TrainNativeMSE runs C++ Mamba NormalBP with lr>0. Packed In|Out|ALog|DSkip stays
// pending until SyncWeightsFromNative.
func TrainNativeMSE[T core.Numeric](l *Layer, input, target *core.Tensor[T], lr float64) (loss float64, err error) {
	if l == nil || l.InProj == nil || l.OutProj == nil || input == nil || target == nil {
		return 0, fmt.Errorf("mamba: TrainNativeMSE nil")
	}
	if lr == 0 {
		return 0, fmt.Errorf("mamba: TrainNativeMSE lr must be > 0")
	}
	if l.InProj.Weights == nil || l.InProj.Weights.Format != quant.FormatNone || !nativecpp.NativeCPUAvailable() {
		return 0, fmt.Errorf("mamba: TrainNativeMSE needs FormatNone + libwelvet")
	}
	lay, err := parseIn(l.Cfg, input)
	if err != nil {
		return 0, err
	}
	packed, err := packMambaNative(l)
	if err != nil {
		return 0, err
	}
	n := lay.batch * lay.t * lay.d
	nw := len(packed)
	xF := core.SliceAsFloat32(input.Data)
	tF := core.SliceAsFloat32(target.Data)
	preF := make([]float32, n)
	postF := make([]float32, n)
	dX := make([]float32, n)
	dW := make([]float32, nw)
	wAfter := make([]float32, nw)
	loss, err = nativecpp.MambaNormalBPF32(
		int32(lay.batch), int32(lay.t), int32(l.Cfg.DModel), int32(l.Cfg.DState), int32(l.Cfg.Expand),
		int32(l.InProj.Weights.DType), nativecpp.CABBackendCPU,
		packed, xF, tF, true, lr,
		preF, postF, dX, dW, wAfter,
	)
	if err != nil {
		return 0, err
	}
	nIn := l.InProj.Weights.Rows * l.InProj.Weights.Cols
	nOut := l.OutProj.Weights.Rows * l.OutProj.Weights.Cols
	inner := l.Cfg.InnerDim()
	l.InProj.EnsureNativeMirror().Capture(wAfter[:nIn], l.InProj.Weights)
	l.OutProj.EnsureNativeMirror().Capture(wAfter[nIn:nIn+nOut], l.OutProj.Weights)
	l.pendingAD = append([]float32(nil), wAfter[nIn+nOut:nIn+nOut+2*inner]...)
	l.adPending = true
	l.ensureMirror().Capture(wAfter[:nIn+nOut], l.InProj.Weights, l.OutProj.Weights)
	return loss, nil
}

func packMambaNative(l *Layer) ([]float32, error) {
	inW, err := l.InProj.NativeMatrixF32()
	if err != nil {
		return nil, err
	}
	outW, err := l.OutProj.NativeMatrixF32()
	if err != nil {
		return nil, err
	}
	inner := l.Cfg.InnerDim()
	out := make([]float32, len(inW)+len(outW)+2*inner)
	copy(out[:len(inW)], inW)
	copy(out[len(inW):len(inW)+len(outW)], outW)
	off := len(inW) + len(outW)
	if l.adPending && len(l.pendingAD) >= 2*inner {
		copy(out[off:], l.pendingAD[:2*inner])
	} else {
		copy(out[off:off+inner], l.ALog)
		copy(out[off+inner:off+2*inner], l.DSkip)
	}
	return out, nil
}

func (l *Layer) SyncWeightsFromNative() error {
	if l == nil {
		return nil
	}
	if l.nativeMir != nil {
		if err := l.nativeMir.SyncToGo(); err != nil {
			return err
		}
	}
	if err := l.InProj.SyncWeightsFromNative(); err != nil {
		return err
	}
	if err := l.OutProj.SyncWeightsFromNative(); err != nil {
		return err
	}
	if l.adPending {
		inner := l.Cfg.InnerDim()
		copy(l.ALog, l.pendingAD[:inner])
		copy(l.DSkip, l.pendingAD[inner:2*inner])
		l.adPending = false
		l.pendingAD = nil
	}
	return nil
}

func (l *Layer) NativeWeightsPending() bool {
	if l == nil {
		return false
	}
	if l.adPending || (l.nativeMir != nil && l.nativeMir.Dirty()) {
		return true
	}
	return l.InProj.NativeWeightsPending() || l.OutProj.NativeWeightsPending()
}

func (l *Layer) ensureMirror() *nativecpp.Mirror {
	if l.nativeMir == nil {
		l.nativeMir = &nativecpp.Mirror{}
	}
	return l.nativeMir
}
