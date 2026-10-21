package softmax

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/nativecpp"
)

// TrainNativeMSE runs C++ Softmax NormalBP (weightless; lr unused for stores).
func TrainNativeMSE[T core.Numeric](l *Layer, input, target *core.Tensor[T], lr float64) (loss float64, err error) {
	if l == nil || input == nil || target == nil {
		return 0, fmt.Errorf("softmax: TrainNativeMSE nil")
	}
	if lr == 0 {
		return 0, fmt.Errorf("softmax: TrainNativeMSE lr must be > 0")
	}
	if !nativecpp.NativeCPUAvailable() {
		return 0, fmt.Errorf("softmax: TrainNativeMSE needs libwelvet")
	}
	n := len(input.Data)
	xF := core.SliceAsFloat32(input.Data)
	tF := core.SliceAsFloat32(target.Data)
	preF := make([]float32, n)
	postF := make([]float32, n)
	dX := make([]float32, n)
	shape := make([]int32, len(input.Shape))
	for i, s := range input.Shape {
		shape[i] = int32(s)
	}
	var mask []byte
	if len(l.Cfg.Mask) > 0 {
		mask = make([]byte, len(l.Cfg.Mask))
		for i, ok := range l.Cfg.Mask {
			if ok {
				mask[i] = 1
			}
		}
	}
	var hier []int32
	if len(l.Cfg.HierarchyLevels) > 0 {
		hier = make([]int32, len(l.Cfg.HierarchyLevels))
		for i, v := range l.Cfg.HierarchyLevels {
			hier[i] = int32(v)
		}
	}
	temp := l.Cfg.Temperature
	if temp <= 0 {
		temp = 1
	}
	loss, err = nativecpp.SoftmaxNormalBPF32(
		int32(n), int32(l.Cfg.Dim), int32(l.Cfg.SeqLen), int32(l.Cfg.Kind),
		int32(l.Cfg.Rows), int32(l.Cfg.Cols), temp, l.Cfg.EntmaxAlpha, 0,
		int32(core.DTypeFloat32), nativecpp.CABBackendCPU,
		int32(len(shape)), shape, mask, hier, xF, tF, true, lr,
		preF, postF, dX,
	)
	return loss, err
}

// SyncWeightsFromNative is a no-op (Softmax is weightless).
func (l *Layer) SyncWeightsFromNative() error { return nil }

// NativeWeightsPending is always false (no stores).
func (l *Layer) NativeWeightsPending() bool { return false }
