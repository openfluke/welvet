package softmax

import (
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/nativecpp"
)

func ForwardNativeCPU[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if l == nil || input == nil || !nativecpp.NativeCPUAvailable() {
		return ForwardCPUTiled(l, input)
	}
	n := len(input.Data)
	xF := core.SliceAsFloat32(input.Data)
	preF := make([]float32, n)
	postF := make([]float32, n)
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
	alpha := l.Cfg.EntmaxAlpha
	if err := nativecpp.SoftmaxForwardF32(
		int32(n), int32(l.Cfg.Dim), int32(l.Cfg.SeqLen), int32(l.Cfg.Kind),
		int32(l.Cfg.Rows), int32(l.Cfg.Cols), temp, alpha, 0,
		int32(core.DTypeFloat32), nativecpp.CABBackendCPU,
		int32(len(shape)), shape, mask, hier, xF, preF, postF,
	); err != nil {
		return ForwardCPUTiled(l, input)
	}
	pre = core.NewTensor[T](input.Shape...)
	post = core.NewTensor[T](input.Shape...)
	core.SliceFromFloat32(preF, pre.Data)
	core.SliceFromFloat32(postF, post.Data)
	return pre, post, nil
}

func BackwardNativeCPU[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if l == nil || !nativecpp.NativeCPUAvailable() {
		return BackwardCPUTiled(l, gradOut, input, pre)
	}
	n := len(input.Data)
	xF := core.SliceAsFloat32(input.Data)
	gF := core.SliceAsFloat32(gradOut.Data)
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
	if _, err := nativecpp.SoftmaxNormalBPF32(
		int32(n), int32(l.Cfg.Dim), int32(l.Cfg.SeqLen), int32(l.Cfg.Kind),
		int32(l.Cfg.Rows), int32(l.Cfg.Cols), temp, l.Cfg.EntmaxAlpha, 0,
		int32(core.DTypeFloat32), nativecpp.CABBackendCPU,
		int32(len(shape)), shape, mask, hier, xF, gF, false, 0,
		preF, postF, dX,
	); err != nil {
		return BackwardCPUTiled(l, gradOut, input, pre)
	}
	gradIn = core.NewTensor[T](input.Shape...)
	core.SliceFromFloat32(dX, gradIn.Data)
	return gradIn, nil, nil
}
