package gdn

import (
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/nativecpp"
)

func ForwardNativeCPU[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	_ = nativecpp.NativeCPUAvailable
	return forwardHost(l, input)
}

func BackwardNativeCPU[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	_ = nativecpp.NativeCPUAvailable
	return BackwardSIMD(l, gradOut, input, pre)
}
