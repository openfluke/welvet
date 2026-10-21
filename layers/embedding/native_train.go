package embedding

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/weights"
)

// TrainNativeMSE runs C++ Embedding NormalBP with lr>0. W_after stays pending
// until SyncWeightsFromNative (for .entity / Go save).
func TrainNativeMSE[T core.Numeric](l *Layer, input, target *core.Tensor[T], lr float64) (loss float64, err error) {
	if l == nil || l.Weights == nil || input == nil || target == nil {
		return 0, fmt.Errorf("embedding: TrainNativeMSE nil")
	}
	if lr == 0 {
		return 0, fmt.Errorf("embedding: TrainNativeMSE lr must be > 0")
	}
	if l.Weights.Format != quant.FormatNone || !nativecpp.NativeCPUAvailable() {
		return 0, fmt.Errorf("embedding: TrainNativeMSE needs FormatNone + libwelvet")
	}
	lay, err := parseLayout(l.Cfg, input)
	if err != nil {
		return 0, err
	}
	w, err := l.NativeMatrixF32()
	if err != nil {
		return 0, err
	}
	nTok := lay.batch * lay.seq
	nOut := nTok * lay.emb
	nw := lay.vocab * lay.emb
	xF := core.SliceAsFloat32(input.Data)
	tF := core.SliceAsFloat32(target.Data)
	preF := make([]float32, nOut)
	postF := make([]float32, nOut)
	dX := make([]float32, nTok*lay.inLast)
	dW := make([]float32, nw)
	wAfter := make([]float32, nw)
	loss, err = nativecpp.EmbeddingNormalBPF32(
		int32(lay.batch), int32(lay.vocab), int32(lay.emb), int32(lay.seq),
		int32(lay.inRank), int32(lay.inLast),
		int32(l.Core.Activation), int32(l.Weights.DType), nativecpp.CABBackendCPU,
		w, xF, tF, true, lr,
		preF, postF, dX, dW, wAfter,
	)
	if err != nil {
		return 0, err
	}
	l.EnsureNativeMirror().Capture(wAfter, l.Weights)
	return loss, nil
}

// SyncWeightsFromNative pulls pending C++ table into Go for .entity save.
func (l *Layer) SyncWeightsFromNative() error {
	if l == nil || l.nativeMir == nil {
		return nil
	}
	return l.nativeMir.SyncToGo()
}

// NativeWeightsPending reports whether SyncWeightsFromNative would write.
func (l *Layer) NativeWeightsPending() bool {
	return l != nil && l.nativeMir != nil && l.nativeMir.Dirty()
}

func (l *Layer) EnsureNativeMirror() *nativecpp.Mirror {
	if l.nativeMir == nil {
		l.nativeMir = &nativecpp.Mirror{}
	}
	return l.nativeMir
}

func (l *Layer) NativeMatrixF32() ([]float32, error) {
	if l.nativeMir != nil {
		return l.nativeMir.PendingOrStore(l.Weights)
	}
	return weights.MatrixF32(l.Weights)
}
