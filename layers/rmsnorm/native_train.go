package rmsnorm

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/weights"
)

// TrainNativeMSE runs C++ RMSNorm NormalBP with lr>0; gamma_after stays pending
// until SyncWeightsFromNative (for .entity / Go save).
func TrainNativeMSE[T core.Numeric](l *Layer, input, target *core.Tensor[T], lr float64) (loss float64, err error) {
	if l == nil || l.Gamma == nil || input == nil || target == nil {
		return 0, fmt.Errorf("rmsnorm: TrainNativeMSE nil")
	}
	if lr == 0 {
		return 0, fmt.Errorf("rmsnorm: TrainNativeMSE lr must be > 0")
	}
	if l.Gamma.Format != quant.FormatNone || !nativecpp.NativeCPUAvailable() {
		return 0, fmt.Errorf("rmsnorm: TrainNativeMSE needs FormatNone + libwelvet")
	}
	lay, err := parseLayout(l.Cfg.Dim, input)
	if err != nil {
		return 0, err
	}
	g, err := l.nativeGamma()
	if err != nil {
		return 0, err
	}
	n := lay.batch * lay.seq * lay.dim
	dim := lay.dim
	xF := core.SliceAsFloat32(input.Data)
	tF := core.SliceAsFloat32(target.Data)
	if len(tF) < n {
		return 0, fmt.Errorf("rmsnorm: target short")
	}
	preF := make([]float32, n)
	postF := make([]float32, n)
	dX := make([]float32, n)
	dW := make([]float32, dim)
	gAfter := make([]float32, dim)
	loss, err = nativecpp.RMSNormNormalBPF32(
		int32(lay.batch), int32(lay.seq), int32(lay.dim), l.Cfg.Eps,
		int32(l.Gamma.DType), nativecpp.CABBackendCPU,
		g, xF, tF, true, lr, preF, postF, dX, dW, gAfter,
	)
	if err != nil {
		return 0, err
	}
	l.ensureMirror().Capture(gAfter, l.Gamma)
	return loss, nil
}

// SyncWeightsFromNative pulls pending C++ gamma into Go for .entity save.
func (l *Layer) SyncWeightsFromNative() error {
	if l == nil || l.nativeMir == nil {
		return nil
	}
	return l.nativeMir.SyncToGo()
}

func (l *Layer) NativeWeightsPending() bool {
	return l != nil && l.nativeMir != nil && l.nativeMir.Dirty()
}

func (l *Layer) ensureMirror() *nativecpp.Mirror {
	if l.nativeMir == nil {
		l.nativeMir = &nativecpp.Mirror{}
	}
	return l.nativeMir
}

func (l *Layer) nativeGamma() ([]float32, error) {
	if l.nativeMir != nil {
		return l.nativeMir.PendingOrStore(l.Gamma)
	}
	return weights.MatrixF32(l.Gamma)
}
