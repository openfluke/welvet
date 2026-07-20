package softmax

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/webgpu"
)

// gpuSupported reports whether Kind maps onto the standard softmax math the
// device kernel implements (max-reduce, exp-sum-reduce, normalize). Gumbel,
// Masked, Sparse and Entmax need per-kind logic the shader does not carry —
// they are host-only on WebGPU (honest error, no silent host fallback).
func gpuSupported(k Kind) bool {
	switch k {
	case KindStandard, KindTemperature, KindGrid, KindHierarchical:
		return true
	default:
		return false
	}
}

// ForwardWebGPU — real device only (no host fake); standard/grid/hierarchical
// Softmax runs on-device (webgpu.Softmax, one workgroup per row/group).
// Gumbel/Masked/Sparse/Entmax are not implemented on-device and error instead
// of silently falling back to host.
func ForwardWebGPU[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("softmax: BackendWebGPU but no device (no host fake)")
	}
	if !gpuSupported(l.Cfg.Kind) {
		return nil, nil, fmt.Errorf("softmax: WebGPU only standard/grid/hierarchical; kind %s host-only", l.Cfg.Kind)
	}
	lay, err := parseLayout(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	xF := core.SliceAsFloat32(input.Data)
	yF := make([]float32, lay.n)
	if err := webgpu.Softmax(xF, yF, lay.rows, lay.cols, float32(lay.temp)); err != nil {
		return nil, nil, err
	}
	y := make([]T, lay.n)
	core.SliceFromFloat32(yF, y)
	pre = core.NewTensor[T](lay.shape...)
	post = core.NewTensor[T](lay.shape...)
	copy(pre.Data, y)
	copy(post.Data, y)
	return pre, post, nil
}

// BackwardWebGPU — dx = (y/T)⊙(dy−⟨dy,y⟩) computed on-device (webgpu.SoftmaxBackward).
// pre must be probabilities from ForwardWebGPU.
func BackwardWebGPU[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("softmax: BackendWebGPU but no device (no host fake)")
	}
	if !gpuSupported(l.Cfg.Kind) {
		return nil, nil, fmt.Errorf("softmax: WebGPU only standard/grid/hierarchical; kind %s host-only", l.Cfg.Kind)
	}
	lay, err := parseLayout(l.Cfg, pre)
	if err != nil {
		lay, err = parseLayout(l.Cfg, gradOut)
		if err != nil {
			return nil, nil, err
		}
	}
	if gradOut == nil || pre == nil || gradOut.Len() < lay.n || pre.Len() < lay.n {
		return nil, nil, fmt.Errorf("softmax: nil/short gradOut/pre")
	}
	gyF := core.SliceAsFloat32(gradOut.Data)
	yF := core.SliceAsFloat32(pre.Data)
	gxF := make([]float32, lay.n)
	if err := webgpu.SoftmaxBackward(gyF, yF, gxF, lay.rows, lay.cols, float32(lay.temp)); err != nil {
		return nil, nil, err
	}
	dx := make([]T, lay.n)
	core.SliceFromFloat32(gxF, dx)
	gradIn = core.NewTensor[T](lay.shape...)
	copy(gradIn.Data, dx)
	return gradIn, nil, nil
}
