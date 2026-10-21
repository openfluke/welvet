package dense

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/weights"
)

// nativeMirror holds pending C++ W_after until SyncWeightsFromNative (e.g. .entity save).
var _ = nativecpp.Mirror{} // keep import stable if methods move

// TrainNativeMSE runs welvet.cpp Dense NormalBP with SGD (lr>0).
// Updated weights stay pending on the layer mirror — call SyncWeightsFromNative
// when you want them in Go Stores (ENTITY export, Go CPU fallback, etc.).
func TrainNativeMSE[T core.Numeric](l *Layer, input, target *core.Tensor[T], lr float64) (loss float64, err error) {
	if l == nil || input == nil || target == nil {
		return 0, fmt.Errorf("dense: TrainNativeMSE nil args")
	}
	if lr == 0 {
		return 0, fmt.Errorf("dense: TrainNativeMSE lr must be > 0")
	}
	if !useNativeCPU(l) {
		return 0, fmt.Errorf("dense: TrainNativeMSE needs FormatNone + libwelvet")
	}
	batch, in, out, err := dims(l, input)
	if err != nil {
		return 0, err
	}
	if len(target.Shape) < 2 || target.Shape[0] != batch || target.Shape[1] != out {
		return 0, fmt.Errorf("dense: TrainNativeMSE target shape want [%d,%d]", batch, out)
	}
	w, err := l.NativeMatrixF32()
	if err != nil {
		return 0, err
	}
	xF := core.SliceAsFloat32(input.Data)
	tF := core.SliceAsFloat32(target.Data)
	preF := make([]float32, batch*out)
	postF := make([]float32, batch*out)
	dX := make([]float32, batch*in)
	dW := make([]float32, out*in)
	wAfter := make([]float32, out*in)
	loss, err = nativecpp.DenseNormalBPF32(
		int32(batch), int32(in), int32(out),
		int32(l.Core.Activation), int32(l.Weights.DType), nativecpp.CABBackendCPU,
		w, l.Weights.Bias, xF, tF, true, lr,
		preF, postF, dX, dW, wAfter,
	)
	if err != nil {
		return 0, err
	}
	l.EnsureNativeMirror().Capture(wAfter, l.Weights)
	return loss, nil
}

// SyncWeightsFromNative pulls pending C++ weights into Go l.Weights for .entity save.
// No-op if nothing is pending. Outside the training loop on purpose.
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

// CommitWeightsFromNative force-commits an explicit buffer (advanced / tests).
func CommitWeightsFromNative(s *weights.Store, wAfter []float32) error {
	return nativecpp.CommitStore(s, wAfter)
}
