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

// BackwardWebGPU — dx/dγ/dβ stay on host: the LayerNorm backward reduction
// needs the same per-token mean/var already produced during the forward pass,
// and welvet's forward WebGPU contract only returns x̂ (not mean/var), so
// backward re-derives them cheaply on host from x̂ itself. Fusing this into a
// second on-device pass is a possible follow-up; for now this keeps a real
// on-device forward (the expensive, batch-scaling side) with an honest host
// backward note in suite reporting ("fwd on-device; bwd host").
func BackwardWebGPU[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("layernorm: BackendWebGPU but no device (no host fake)")
	}
	return backwardHost(l, gradOut, input, pre, false)
}
