package rnn

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/quant"
)

// TrainNativeMSE runs C++ RNN NormalBP with lr>0. Packed IH|HH|bias W_after stays
// pending (matrices on child Dense mirrors; bias on layer) until SyncWeightsFromNative.
func TrainNativeMSE[T core.Numeric](l *Layer, input, target *core.Tensor[T], lr float64) (loss float64, err error) {
	if l == nil || l.IH == nil || l.HH == nil || input == nil || target == nil {
		return 0, fmt.Errorf("rnn: TrainNativeMSE nil")
	}
	if lr == 0 {
		return 0, fmt.Errorf("rnn: TrainNativeMSE lr must be > 0")
	}
	if l.IH.Weights == nil || l.IH.Weights.Format != quant.FormatNone || !nativecpp.NativeCPUAvailable() {
		return 0, fmt.Errorf("rnn: TrainNativeMSE needs FormatNone + libwelvet")
	}
	lay, err := parseLayout(l.Cfg, input)
	if err != nil {
		return 0, err
	}
	packed, err := packRNNNative(l)
	if err != nil {
		return 0, err
	}
	nOut := lay.batch * lay.seq * lay.hid
	nIn := lay.batch * lay.seq * lay.in
	nw := len(packed)
	xF := core.SliceAsFloat32(input.Data)
	tF := core.SliceAsFloat32(target.Data)
	preF := make([]float32, nOut)
	postF := make([]float32, nOut)
	dX := make([]float32, nIn)
	dW := make([]float32, nw)
	wAfter := make([]float32, nw)
	loss, err = nativecpp.RNNNormalBPF32(
		int32(lay.batch), int32(lay.in), int32(lay.hid), int32(lay.seq),
		int32(l.IH.Weights.DType), nativecpp.CABBackendCPU,
		packed, xF, tF, true, lr,
		preF, postF, dX, dW, wAfter,
	)
	if err != nil {
		return 0, err
	}
	captureRNNAfter(l, wAfter)
	return loss, nil
}

func packRNNNative(l *Layer) ([]float32, error) {
	ih, err := l.IH.NativeMatrixF32()
	if err != nil {
		return nil, err
	}
	hh, err := l.HH.NativeMatrixF32()
	if err != nil {
		return nil, err
	}
	h := l.Cfg.HiddenSize
	out := make([]float32, len(ih)+len(hh)+h)
	copy(out[:len(ih)], ih)
	copy(out[len(ih):len(ih)+len(hh)], hh)
	if l.biasPending && len(l.pendingBias) >= h {
		copy(out[len(ih)+len(hh):], l.pendingBias[:h])
	} else {
		for i := 0; i < h; i++ {
			if i < len(l.IH.Weights.Bias) {
				out[len(ih)+len(hh)+i] = float32(l.IH.Weights.Bias[i])
			}
		}
	}
	return out, nil
}

func captureRNNAfter(l *Layer, wAfter []float32) {
	ihN := l.IH.Weights.Rows * l.IH.Weights.Cols
	hhN := l.HH.Weights.Rows * l.HH.Weights.Cols
	h := l.Cfg.HiddenSize
	l.IH.EnsureNativeMirror().Capture(wAfter[:ihN], l.IH.Weights)
	l.HH.EnsureNativeMirror().Capture(wAfter[ihN:ihN+hhN], l.HH.Weights)
	l.pendingBias = append([]float32(nil), wAfter[ihN+hhN:ihN+hhN+h]...)
	l.biasPending = true
	l.ensureMirror().Capture(wAfter[:ihN+hhN], l.IH.Weights, l.HH.Weights)
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
	if err := l.IH.SyncWeightsFromNative(); err != nil {
		return err
	}
	if err := l.HH.SyncWeightsFromNative(); err != nil {
		return err
	}
	if l.biasPending && l.IH != nil && l.IH.Weights != nil {
		for i := range l.pendingBias {
			if i < len(l.IH.Weights.Bias) {
				l.IH.Weights.Bias[i] = float64(l.pendingBias[i])
			}
		}
		l.biasPending = false
		l.pendingBias = nil
	}
	return nil
}

func (l *Layer) NativeWeightsPending() bool {
	if l == nil {
		return false
	}
	if l.biasPending || (l.nativeMir != nil && l.nativeMir.Dirty()) {
		return true
	}
	return l.IH.NativeWeightsPending() || l.HH.NativeWeightsPending()
}

func (l *Layer) ensureMirror() *nativecpp.Mirror {
	if l.nativeMir == nil {
		l.nativeMir = &nativecpp.Mirror{}
	}
	return l.nativeMir
}
