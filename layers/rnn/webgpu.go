package rnn

import (
	"fmt"
	"math"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/webgpu"
)

func gpuRNNOK(l *Layer) bool {
	if l == nil || l.IH == nil || l.HH == nil {
		return false
	}
	if l.IH.Weights.Format != quant.FormatNone || l.HH.Weights.Format != quant.FormatNone {
		return false
	}
	if _, err := l.IH.Weights.GPUWireF32(); err != nil {
		return false
	}
	if _, err := l.HH.Weights.GPUWireF32(); err != nil {
		return false
	}
	return true
}

func packRNNWeightsF32(l *Layer) ([]float32, error) {
	ihW, err := l.IH.Weights.GPUWireF32()
	if err != nil {
		return nil, fmt.Errorf("rnn IH weights: %w", err)
	}
	hhW, err := l.HH.Weights.GPUWireF32()
	if err != nil {
		return nil, fmt.Errorf("rnn HH weights: %w", err)
	}
	h, in := l.Cfg.HiddenSize, l.Cfg.InputSize
	ihN, hhN := h*in, h*h
	out := make([]float32, ihN+hhN+h)
	copy(out[:ihN], ihW)
	copy(out[ihN:ihN+hhN], hhW)
	for i := 0; i < h; i++ {
		out[ihN+hhN+i] = float32(l.IH.Weights.Bias[i])
	}
	return out, nil
}

// ForwardWebGPU — fused RNN recurrence on device (FormatNone f32).
func ForwardWebGPU[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("rnn: BackendWebGPU but no device (no host fake)")
	}
	if !gpuRNNOK(l) {
		return forwardViaDense(l, input)
	}
	lay, err := parseLayout(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	weights, err := packRNNWeightsF32(l)
	if err != nil {
		return nil, nil, err
	}

	pre = core.NewTensor[T](lay.batch, lay.seq, lay.hid)
	post = core.NewTensor[T](lay.batch, lay.seq, lay.hid)
	inF := core.SliceAsFloat32(input.Data)
	preF := make([]float32, lay.batch*lay.seq*lay.hid)
	postF := make([]float32, lay.batch*lay.seq*lay.hid)
	if err := webgpu.RNNForwardSeq(inF, weights, preF, postF, lay.batch, lay.seq, lay.in, lay.hid); err != nil {
		return nil, nil, err
	}
	core.SliceFromFloat32(preF, pre.Data)
	core.SliceFromFloat32(postF, post.Data)
	return pre, post, nil
}

// BackwardWebGPU — BPTT with on-device step DX/DW (FormatNone f32).
func BackwardWebGPU[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("rnn: BackendWebGPU but no device (no host fake)")
	}
	if !gpuRNNOK(l) {
		return backwardViaDense(l, gradOut, input, pre)
	}
	lay, err := parseLayout(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	if gradOut == nil || pre == nil {
		return nil, nil, fmt.Errorf("rnn: nil gradOut/pre")
	}
	weights, err := packRNNWeightsF32(l)
	if err != nil {
		return nil, nil, err
	}

	tileSize := l.Exec.TileSize
	if tileSize <= 0 {
		tileSize = l.Core.TileSize
	}
	if tileSize <= 0 {
		tileSize = 64
	}

	postF := make([]float32, lay.batch*lay.seq*lay.hid)
	for i := range postF {
		postF[i] = float32(math.Tanh(core.AsFloat64(pre.Data[i])))
	}

	gradOutF := core.SliceAsFloat32(gradOut.Data)
	inF := core.SliceAsFloat32(input.Data)
	gradInF := make([]float32, lay.batch*lay.seq*lay.in)
	gradWF := make([]float32, len(weights))
	if err := webgpu.RNNBackwardSeq(gradOutF, inF, postF, weights, gradInF, gradWF, lay.batch, lay.seq, lay.in, lay.hid, tileSize); err != nil {
		return nil, nil, err
	}

	gradIn = core.NewTensor[T](lay.batch, lay.seq, lay.in)
	core.SliceFromFloat32(gradInF, gradIn.Data)
	gradW = core.NewTensor[T](1, l.GradWSize())
	core.SliceFromFloat32(gradWF, gradW.Data)
	return gradIn, gradW, nil
}
