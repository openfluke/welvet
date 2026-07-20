package rmsnorm

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/webgpu"
)

// ForwardWebGPU — real device only (no host fake); RMS ALU runs on-device
// (webgpu.RMSNorm, one workgroup per token).
func ForwardWebGPU[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("rmsnorm: BackendWebGPU but no device (no host fake)")
	}
	lay, err := parseLayout(l.Cfg.Dim, input)
	if err != nil {
		return nil, nil, err
	}
	g, err := l.Gamma.GPUWireF32()
	if err != nil {
		return nil, nil, fmt.Errorf("rmsnorm gamma: %w", err)
	}
	dim := l.Cfg.Dim
	nTok := tokens(lay)

	xF := core.SliceAsFloat32(input.Data)
	xHatF := make([]float32, nTok*dim)
	yF := make([]float32, nTok*dim)
	if err := webgpu.RMSNormXHat(xF, g, xHatF, yF, nTok, dim, float32(l.Cfg.Eps)); err != nil {
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

// BackwardWebGPU — dx and dGamma computed on-device (webgpu.RMSNormBackward).
// pre must be x̂ = x*invRMS from ForwardWebGPU (before γ).
func BackwardWebGPU[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("rmsnorm: BackendWebGPU but no device (no host fake)")
	}
	lay, err := parseLayout(l.Cfg.Dim, input)
	if err != nil {
		return nil, nil, err
	}
	g, err := l.Gamma.GPUWireF32()
	if err != nil {
		return nil, nil, fmt.Errorf("rmsnorm gamma: %w", err)
	}
	dim := l.Cfg.Dim
	nTok := tokens(lay)
	if pre == nil || pre.Len() < nTok*dim {
		return nil, nil, fmt.Errorf("rmsnorm: pre (x̂) missing")
	}

	xF := core.SliceAsFloat32(input.Data)
	xHatF := core.SliceAsFloat32(pre.Data)
	dyF := core.SliceAsFloat32(gradOut.Data)
	dxF := make([]float32, nTok*dim)
	dGammaF := make([]float32, dim)

	if err := webgpu.RMSNormBackward(dyF, xF, xHatF, g, dxF, dGammaF, nTok, dim, float32(l.Cfg.Eps)); err != nil {
		return nil, nil, err
	}

	dx := make([]T, nTok*dim)
	core.SliceFromFloat32(dxF, dx)
	gradIn = reshapeOut(dx, lay)
	gradW = core.NewTensor[T](1, dim)
	core.SliceFromFloat32(dGammaF, gradW.Data)
	return gradIn, gradW, nil
}
