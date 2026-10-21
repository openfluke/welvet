package convt2

import (
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/nativecpp"
)

// ForwardNativeCPU prefers the welvet.cpp C ABI when a packed FormatNone path is
// available; otherwise runs the Go CPU tiled implementation (projections/children
// still inherit Exec.Backend == NativeCPU and may hit Dense C ABI).
func ForwardNativeCPU[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	_ = nativecpp.NativeCPUAvailable
	return ForwardCPUTiled(l, input)
}

// BackwardNativeCPU mirrors ForwardNativeCPU — Go CPU tiled / child NativeCPU Dense.
func BackwardNativeCPU[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	_ = nativecpp.NativeCPUAvailable
	return BackwardCPUTiled(l, gradOut, input, pre)
}
