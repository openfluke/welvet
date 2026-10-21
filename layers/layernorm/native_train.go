package layernorm

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/weights"
)

// TrainNativeMSE runs C++ LayerNorm NormalBP with lr>0; γ/β stay pending until Sync.
func TrainNativeMSE[T core.Numeric](l *Layer, input, target *core.Tensor[T], lr float64) (loss float64, err error) {
	if l == nil || l.Gamma == nil || l.Beta == nil || input == nil || target == nil {
		return 0, fmt.Errorf("layernorm: TrainNativeMSE nil")
	}
	if lr == 0 {
		return 0, fmt.Errorf("layernorm: TrainNativeMSE lr must be > 0")
	}
	if l.Gamma.Format != quant.FormatNone || !nativecpp.NativeCPUAvailable() {
		return 0, fmt.Errorf("layernorm: TrainNativeMSE needs FormatNone + libwelvet")
	}
	lay, err := parseLayout(l.Cfg.Dim, input)
	if err != nil {
		return 0, err
	}
	g, err := pendingOr(l, l.Gamma)
	if err != nil {
		return 0, err
	}
	b, err := pendingOr(l, l.Beta)
	if err != nil {
		return 0, err
	}
	n := lay.batch * lay.seq * lay.dim
	dim := lay.dim
	xF := core.SliceAsFloat32(input.Data)
	tF := core.SliceAsFloat32(target.Data)
	preF := make([]float32, n)
	postF := make([]float32, n)
	dX := make([]float32, n)
	dW := make([]float32, dim*2)
	gAfter := make([]float32, dim)
	bAfter := make([]float32, dim)
	loss, err = nativecpp.LayerNormNormalBPF32(
		int32(lay.batch), int32(lay.seq), int32(lay.dim), l.Cfg.Eps,
		int32(l.Gamma.DType), nativecpp.CABBackendCPU,
		g, b, xF, tF, true, lr, preF, postF, dX, dW, gAfter, bAfter,
	)
	if err != nil {
		return 0, err
	}
	after := append(append([]float32{}, gAfter...), bAfter...)
	l.ensureMirror().Capture(after, l.Gamma, l.Beta)
	return loss, nil
}

func pendingOr(l *Layer, s *weights.Store) ([]float32, error) {
	if l.nativeMir != nil {
		return l.nativeMir.PendingOrStore(s)
	}
	return weights.MatrixF32(s)
}

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
