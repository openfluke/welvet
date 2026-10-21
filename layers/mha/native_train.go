package mha

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/quant"
)

// TrainNativeMSE runs C++ MHA NormalBP with lr>0. Q|K|V|O W_after stays pending
// until SyncWeightsFromNative (for .entity / Go save).
func TrainNativeMSE[T core.Numeric](l *Layer, input, target *core.Tensor[T], lr float64) (loss float64, err error) {
	if l == nil || l.Q == nil || l.K == nil || l.V == nil || l.O == nil || input == nil || target == nil {
		return 0, fmt.Errorf("mha: TrainNativeMSE nil")
	}
	if lr == 0 {
		return 0, fmt.Errorf("mha: TrainNativeMSE lr must be > 0")
	}
	if l.Q.Weights == nil || l.Q.Weights.Format != quant.FormatNone || !nativecpp.NativeCPUAvailable() {
		return 0, fmt.Errorf("mha: TrainNativeMSE needs FormatNone + libwelvet")
	}
	if l.Cfg.Mode != ModeSelf {
		return 0, fmt.Errorf("mha: TrainNativeMSE C ABI is ModeSelf only")
	}
	lay, err := parseLayout(l.Cfg.DModel, input)
	if err != nil {
		return 0, err
	}
	cfg := l.Cfg
	maxSeq := cfg.MaxSeqLen
	if maxSeq <= 0 {
		maxSeq = 512
	}
	q, err := l.Q.NativeMatrixF32()
	if err != nil {
		return 0, err
	}
	k, err := l.K.NativeMatrixF32()
	if err != nil {
		return 0, err
	}
	v, err := l.V.NativeMatrixF32()
	if err != nil {
		return 0, err
	}
	o, err := l.O.NativeMatrixF32()
	if err != nil {
		return 0, err
	}
	n := lay.batch * lay.seqLen * lay.dModel
	nw := len(q) + len(k) + len(v) + len(o)
	xF := core.SliceAsFloat32(input.Data)
	tF := core.SliceAsFloat32(target.Data)
	preF := make([]float32, n)
	postF := make([]float32, n)
	dX := make([]float32, n)
	dW := make([]float32, nw)
	wAfter := make([]float32, nw)
	loss, err = nativecpp.MHANormalBPF32(
		int32(lay.batch), int32(lay.seqLen), int32(cfg.DModel),
		int32(cfg.NumHeads), int32(cfg.NumKVHeads), int32(cfg.HeadDim), int32(maxSeq),
		int32(cfg.Mask), int32(cfg.Pos),
		int32(l.Q.Weights.DType), nativecpp.CABBackendCPU,
		q, k, v, o, xF, tF, true, lr,
		preF, postF, dX, dW, wAfter,
	)
	if err != nil {
		return 0, err
	}
	off := 0
	l.Q.EnsureNativeMirror().Capture(wAfter[off:off+len(q)], l.Q.Weights)
	off += len(q)
	l.K.EnsureNativeMirror().Capture(wAfter[off:off+len(k)], l.K.Weights)
	off += len(k)
	l.V.EnsureNativeMirror().Capture(wAfter[off:off+len(v)], l.V.Weights)
	off += len(v)
	l.O.EnsureNativeMirror().Capture(wAfter[off:off+len(o)], l.O.Weights)
	l.ensureMirror().Capture(wAfter, l.Q.Weights, l.K.Weights, l.V.Weights, l.O.Weights)
	return loss, nil
}

// SyncWeightsFromNative pulls pending C++ Q/K/V/O into Go for .entity save.
func (l *Layer) SyncWeightsFromNative() error {
	if l == nil {
		return nil
	}
	if l.nativeMir != nil {
		if err := l.nativeMir.SyncToGo(); err != nil {
			return err
		}
	}
	if err := l.Q.SyncWeightsFromNative(); err != nil {
		return err
	}
	if err := l.K.SyncWeightsFromNative(); err != nil {
		return err
	}
	if err := l.V.SyncWeightsFromNative(); err != nil {
		return err
	}
	return l.O.SyncWeightsFromNative()
}

// NativeWeightsPending reports whether SyncWeightsFromNative would write.
func (l *Layer) NativeWeightsPending() bool {
	if l == nil {
		return false
	}
	if l.nativeMir != nil && l.nativeMir.Dirty() {
		return true
	}
	return l.Q.NativeWeightsPending() || l.K.NativeWeightsPending() ||
		l.V.NativeWeightsPending() || l.O.NativeWeightsPending()
}

func (l *Layer) ensureMirror() *nativecpp.Mirror {
	if l.nativeMir == nil {
		l.nativeMir = &nativecpp.Mirror{}
	}
	return l.nativeMir
}
