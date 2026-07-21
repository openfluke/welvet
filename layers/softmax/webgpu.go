package softmax

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/webgpu"
)

// gpuSupported reports whether Kind has an on-device forward/backward kernel.
func gpuSupported(k Kind) bool {
	switch k {
	case KindStandard, KindTemperature, KindGrid, KindHierarchical,
		KindGumbel, KindMasked, KindSparse, KindEntmax:
		return true
	default:
		return false
	}
}

func kindToGPUType(k Kind) uint32 {
	switch k {
	case KindStandard:
		return webgpu.SoftmaxTypeStandard
	case KindGrid:
		return webgpu.SoftmaxTypeGrid
	case KindHierarchical:
		return webgpu.SoftmaxTypeHierarchical
	case KindTemperature:
		return webgpu.SoftmaxTypeTemperature
	case KindGumbel:
		return webgpu.SoftmaxTypeGumbel
	case KindMasked:
		return webgpu.SoftmaxTypeMasked
	case KindSparse:
		return webgpu.SoftmaxTypeSparse
	case KindEntmax:
		return webgpu.SoftmaxTypeEntmax
	default:
		return webgpu.SoftmaxTypeStandard
	}
}

func maskF32FromBool(mask []bool, n int) []float32 {
	if len(mask) == 0 {
		return nil
	}
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		if i < len(mask) && mask[i] {
			out[i] = 1
		}
	}
	return out
}

func entmaxAlphaF32(cfg Config) float32 {
	alpha := float32(cfg.EntmaxAlpha)
	if alpha == 0 {
		alpha = 1.5
	}
	return alpha
}

// ForwardWebGPU — real device only (no host fake).
func ForwardWebGPU[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("softmax: BackendWebGPU but no device (no host fake)")
	}
	if !gpuSupported(l.Cfg.Kind) {
		return nil, nil, fmt.Errorf("softmax: WebGPU unsupported kind %s", l.Cfg.Kind)
	}
	lay, err := parseLayout(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	xF := core.SliceAsFloat32(input.Data)
	yF := make([]float32, lay.n)
	maskF := maskF32FromBool(l.Cfg.Mask, lay.n)
	kind := kindToGPUType(l.Cfg.Kind)
	if err := webgpu.SoftmaxEx(xF, yF, maskF, lay.rows, lay.cols, float32(lay.temp), kind, 1, entmaxAlphaF32(l.Cfg)); err != nil {
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
		return nil, nil, fmt.Errorf("softmax: WebGPU unsupported kind %s", l.Cfg.Kind)
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
	if l.Cfg.Kind == KindMasked {
		for i := 0; i < lay.n; i++ {
			if i < len(l.Cfg.Mask) && !l.Cfg.Mask[i] {
				gyF[i] = 0
			}
		}
	}
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
