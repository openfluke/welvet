package lstm

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/quant"
)

// TrainNativeMSE runs C++ LSTM NormalBP with lr>0. Packed i|f|g|o stays pending
// until SyncWeightsFromNative.
func TrainNativeMSE[T core.Numeric](l *Layer, input, target *core.Tensor[T], lr float64) (loss float64, err error) {
	if l == nil || l.I == nil || input == nil || target == nil {
		return 0, fmt.Errorf("lstm: TrainNativeMSE nil")
	}
	if lr == 0 {
		return 0, fmt.Errorf("lstm: TrainNativeMSE lr must be > 0")
	}
	if l.I.IH == nil || l.I.IH.Weights == nil || l.I.IH.Weights.Format != quant.FormatNone || !nativecpp.NativeCPUAvailable() {
		return 0, fmt.Errorf("lstm: TrainNativeMSE needs FormatNone + libwelvet")
	}
	lay, err := parseLayout(l.Cfg, input)
	if err != nil {
		return 0, err
	}
	packed, err := packLSTMNative(l)
	if err != nil {
		return 0, err
	}
	nOut := lay.batch * lay.seq * lay.hid
	nIn := lay.batch * lay.seq * lay.in
	nPre := lay.batch * lay.seq * 5 * lay.hid
	nw := len(packed)
	xF := core.SliceAsFloat32(input.Data)
	tF := core.SliceAsFloat32(target.Data)
	preF := make([]float32, nPre)
	postF := make([]float32, nOut)
	dX := make([]float32, nIn)
	dW := make([]float32, nw)
	wAfter := make([]float32, nw)
	loss, err = nativecpp.LSTMNormalBPF32(
		int32(lay.batch), int32(lay.in), int32(lay.hid), int32(lay.seq),
		int32(l.I.IH.Weights.DType), nativecpp.CABBackendCPU,
		packed, xF, tF, true, lr,
		preF, postF, dX, dW, wAfter,
	)
	if err != nil {
		return 0, err
	}
	captureLSTMAfter(l, wAfter)
	return loss, nil
}

func packLSTMNative(l *Layer) ([]float32, error) {
	h, in := l.Cfg.HiddenSize, l.Cfg.InputSize
	gateN := l.Cfg.GateSize()
	out := make([]float32, l.Cfg.WeightCount())
	for gi, g := range l.gates() {
		ih, err := g.IH.NativeMatrixF32()
		if err != nil {
			return nil, err
		}
		hh, err := g.HH.NativeMatrixF32()
		if err != nil {
			return nil, err
		}
		ihN, hhN := h*in, h*h
		base := gi * gateN
		copy(out[base:base+ihN], ih)
		copy(out[base+ihN:base+ihN+hhN], hh)
		if l.biasPending && len(l.pendingBias) >= (gi+1)*h {
			copy(out[base+ihN+hhN:base+gateN], l.pendingBias[gi*h:(gi+1)*h])
		} else {
			for i := 0; i < h; i++ {
				if i < len(g.IH.Weights.Bias) {
					out[base+ihN+hhN+i] = float32(g.IH.Weights.Bias[i])
				}
			}
		}
	}
	return out, nil
}

func captureLSTMAfter(l *Layer, wAfter []float32) {
	h, in := l.Cfg.HiddenSize, l.Cfg.InputSize
	gateN := l.Cfg.GateSize()
	ihN, hhN := h*in, h*h
	bias := make([]float32, 4*h)
	matParts := make([]float32, 0, 4*(ihN+hhN))
	for gi, g := range l.gates() {
		base := gi * gateN
		g.IH.EnsureNativeMirror().Capture(wAfter[base:base+ihN], g.IH.Weights)
		g.HH.EnsureNativeMirror().Capture(wAfter[base+ihN:base+ihN+hhN], g.HH.Weights)
		copy(bias[gi*h:(gi+1)*h], wAfter[base+ihN+hhN:base+gateN])
		matParts = append(matParts, wAfter[base:base+ihN+hhN]...)
	}
	l.pendingBias = bias
	l.biasPending = true
	l.ensureMirror().Capture(matParts,
		l.I.IH.Weights, l.I.HH.Weights,
		l.F.IH.Weights, l.F.HH.Weights,
		l.G.IH.Weights, l.G.HH.Weights,
		l.O.IH.Weights, l.O.HH.Weights,
	)
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
	h := l.Cfg.HiddenSize
	for gi, g := range l.gates() {
		if err := g.IH.SyncWeightsFromNative(); err != nil {
			return err
		}
		if err := g.HH.SyncWeightsFromNative(); err != nil {
			return err
		}
		if l.biasPending && g.IH != nil && g.IH.Weights != nil {
			for i := 0; i < h; i++ {
				if i < len(g.IH.Weights.Bias) && gi*h+i < len(l.pendingBias) {
					g.IH.Weights.Bias[i] = float64(l.pendingBias[gi*h+i])
				}
			}
		}
	}
	l.biasPending = false
	l.pendingBias = nil
	return nil
}

func (l *Layer) NativeWeightsPending() bool {
	if l == nil {
		return false
	}
	if l.biasPending || (l.nativeMir != nil && l.nativeMir.Dirty()) {
		return true
	}
	for _, g := range l.gates() {
		if g.IH.NativeWeightsPending() || g.HH.NativeWeightsPending() {
			return true
		}
	}
	return false
}

func (l *Layer) ensureMirror() *nativecpp.Mirror {
	if l.nativeMir == nil {
		l.nativeMir = &nativecpp.Mirror{}
	}
	return l.nativeMir
}
