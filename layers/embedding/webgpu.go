package embedding

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/webgpu"
)

// ForwardWebGPU — on-device gather from the embedding table (real WebGPU only).
func ForwardWebGPU[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("embedding: BackendWebGPU but no device (no host fake)")
	}
	lay, err := parseLayout(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	w, err := l.Weights.GPUWireF32()
	if err != nil {
		return nil, nil, fmt.Errorf("embedding: WebGPU weights: %w", err)
	}
	shape := outShape(lay)
	pre = core.NewTensor[T](shape...)
	outF := make([]float32, lay.nTok*lay.emb)
	idx := tokenIndices(input, lay)
	if err := webgpu.EmbeddingGather(idx, w, outF, lay.vocab, lay.emb, lay.nTok); err != nil {
		return nil, nil, err
	}
	core.SliceFromFloat32(outF, pre.Data)
	n := lay.nTok * lay.emb
	if l.Core.Activation != core.ActivationLinear {
		for i := 0; i < n; i++ {
			pre.Data[i] = core.Activate(pre.Data[i], l.Core.Activation)
		}
	}
	post = pre
	return pre, post, nil
}

// BackwardWebGPU — on-device scatter into dW; gradIn is zeros (no grad through discrete IDs).
func BackwardWebGPU[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	_ = pre
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("embedding: BackendWebGPU but no device (no host fake)")
	}
	lay, err := parseLayout(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	if gradOut == nil || gradOut.Len() < lay.nTok*lay.emb {
		return nil, nil, fmt.Errorf("embedding: gradOut short")
	}
	goF := core.SliceAsFloat32(gradOut.Data[:lay.nTok*lay.emb])
	idx := tokenIndices(input, lay)
	dwF := make([]float32, lay.vocab*lay.emb)
	if err := webgpu.EmbeddingScatter(idx, goF, dwF, lay.vocab, lay.emb, lay.nTok); err != nil {
		return nil, nil, err
	}
	gradW = core.NewTensor[T](lay.vocab, lay.emb)
	core.SliceFromFloat32(dwF, gradW.Data)
	gradIn = core.NewTensor[T](input.Shape...)
	return gradIn, gradW, nil
}

func tokenIndices[T core.Numeric](input *core.Tensor[T], lay layout) []uint32 {
	idx := make([]uint32, lay.nTok)
	for t := 0; t < lay.nTok; t++ {
		idx[t] = uint32(tokenID(input, lay, t))
	}
	return idx
}
