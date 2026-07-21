package layernorm

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/webgpu"
)

// ForwardWebGPU — real device only (no host fake); LN ALU runs on-device
// (webgpu.LayerNorm, one workgroup per token: mean/var/affine fused).
func ForwardWebGPU[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("layernorm: BackendWebGPU but no device (no host fake)")
	}
	lay, err := parseLayout(l.Cfg.Dim, input)
	if err != nil {
		return nil, nil, err
	}
	g, err := l.Gamma.GPUWireF32()
	if err != nil {
		return nil, nil, fmt.Errorf("layernorm gamma: %w", err)
	}
	b, err := l.Beta.GPUWireF32()
	if err != nil {
		return nil, nil, fmt.Errorf("layernorm beta: %w", err)
	}
	dim := l.Cfg.Dim
	nTok := tokens(lay)

	xF := core.SliceAsFloat32(input.Data)
	xHatF := make([]float32, nTok*dim)
	yF := make([]float32, nTok*dim)
	if err := webgpu.LayerNormXHat(xF, g, b, xHatF, yF, nTok, dim, float32(l.Cfg.Eps)); err != nil {
		return nil, nil, err
	}

	xHat := make([]T, nTok*dim)
	y := make([]T, nTok*dim)
	core.SliceFromFloat32(xHatF, xHat)
	core.SliceFromFloat32(yF, y)
	pre = reshapeOut(xHat, lay)
	post = reshapeOut(y, lay)
	return pre, post, nil
}

// BackwardWebGPU — dx, dGamma, and dBeta computed on-device (webgpu.LayerNormBackward).
// pre must be x̂ from ForwardWebGPU (before γ,β affine).
func BackwardWebGPU[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("layernorm: BackendWebGPU but no device (no host fake)")
	}
	lay, err := parseLayout(l.Cfg.Dim, input)
	if err != nil {
		return nil, nil, err
	}
	g, err := l.Gamma.GPUWireF32()
	if err != nil {
		return nil, nil, fmt.Errorf("layernorm gamma: %w", err)
	}
	dim := l.Cfg.Dim
	nTok := tokens(lay)
	if pre == nil || pre.Len() < nTok*dim {
		return nil, nil, fmt.Errorf("layernorm: pre (x̂) missing")
	}

	xF := core.SliceAsFloat32(input.Data)
	xHatF := core.SliceAsFloat32(pre.Data)
	dyF := core.SliceAsFloat32(gradOut.Data)
	dxF := make([]float32, nTok*dim)
	dGammaF := make([]float32, dim)
	dBetaF := make([]float32, dim)

	if err := webgpu.LayerNormBackward(dyF, xF, xHatF, g, dxF, dGammaF, dBetaF, nTok, dim, float32(l.Cfg.Eps)); err != nil {
		return nil, nil, err
	}

	dx := make([]T, nTok*dim)
	core.SliceFromFloat32(dxF, dx)
	gradIn = reshapeOut(dx, lay)
	gradW = core.NewTensor[T](1, 2*dim)
	core.SliceFromFloat32(dGammaF, gradW.Data[:dim])
	core.SliceFromFloat32(dBetaF, gradW.Data[dim:])
	return gradIn, gradW, nil
}
