package sequential

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/weights"
)

// TrainNativeMSE runs C++ Sequential NormalBP (Dense children pack) with lr>0.
// Child W_after stays pending until SyncWeightsFromNative.
func TrainNativeMSE[T core.Numeric](l *Layer, input, target *core.Tensor[T], lr float64) (loss float64, err error) {
	if l == nil || input == nil || target == nil {
		return 0, fmt.Errorf("sequential: TrainNativeMSE nil")
	}
	if lr == 0 {
		return 0, fmt.Errorf("sequential: TrainNativeMSE lr must be > 0")
	}
	if len(l.Ops) > 0 {
		return 0, fmt.Errorf("sequential: TrainNativeMSE needs Dense Children (not mixed Ops)")
	}
	if len(l.Children) == 0 || l.Children[0] == nil || l.Children[0].Weights == nil {
		return 0, fmt.Errorf("sequential: TrainNativeMSE nil children")
	}
	if l.Children[0].Weights.Format != quant.FormatNone || !nativecpp.NativeCPUAvailable() {
		return 0, fmt.Errorf("sequential: TrainNativeMSE needs FormatNone + libwelvet")
	}
	lay, err := parseLayout(l.Cfg.Dim, input)
	if err != nil {
		return 0, err
	}
	packed, stores, err := packDenseChildren(l.Children)
	if err != nil {
		return 0, err
	}
	dim, depth := l.Cfg.Dim, len(l.Children)
	n := lay.batch * lay.seq * dim
	nw := len(packed)
	xF := core.SliceAsFloat32(input.Data)
	tF := core.SliceAsFloat32(target.Data)
	preF := make([]float32, n)
	postF := make([]float32, n)
	dX := make([]float32, n)
	dW := make([]float32, nw)
	wAfter := make([]float32, nw)
	childAct := int32(l.Children[0].Core.Activation)
	loss, err = nativecpp.SequentialNormalBPF32(
		int32(lay.batch), int32(lay.seq), int32(dim), int32(depth), childAct,
		int32(l.Children[0].Weights.DType), nativecpp.CABBackendCPU,
		packed, xF, tF, true, lr,
		preF, postF, dX, dW, wAfter,
	)
	if err != nil {
		return 0, err
	}
	off := 0
	for _, ch := range l.Children {
		nW := ch.Weights.Rows * ch.Weights.Cols
		ch.EnsureNativeMirror().Capture(wAfter[off:off+nW], ch.Weights)
		off += nW
	}
	l.ensureMirror().Capture(wAfter, stores...)
	return loss, nil
}

func packDenseChildren(children []*dense.Layer) ([]float32, []*weights.Store, error) {
	var stores []*weights.Store
	var parts [][]float32
	total := 0
	for i, ch := range children {
		if ch == nil || ch.Weights == nil {
			return nil, nil, fmt.Errorf("sequential: nil child %d", i)
		}
		w, err := ch.NativeMatrixF32()
		if err != nil {
			return nil, nil, err
		}
		parts = append(parts, w)
		stores = append(stores, ch.Weights)
		total += len(w)
	}
	out := make([]float32, total)
	off := 0
	for _, p := range parts {
		copy(out[off:], p)
		off += len(p)
	}
	return out, stores, nil
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
	for _, ch := range l.Children {
		if ch == nil {
			continue
		}
		if err := ch.SyncWeightsFromNative(); err != nil {
			return err
		}
	}
	return nil
}

func (l *Layer) NativeWeightsPending() bool {
	if l == nil {
		return false
	}
	if l.nativeMir != nil && l.nativeMir.Dirty() {
		return true
	}
	for _, ch := range l.Children {
		if ch != nil && ch.NativeWeightsPending() {
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
