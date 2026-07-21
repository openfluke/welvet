package lstm

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/webgpu"
)

func gpuLSTMOK(l *Layer) bool {
	if l == nil || l.I == nil {
		return false
	}
	for _, g := range l.gates() {
		if g == nil || g.IH == nil || g.HH == nil {
			return false
		}
		if g.IH.Weights.Format != quant.FormatNone || g.HH.Weights.Format != quant.FormatNone {
			return false
		}
		if _, err := g.IH.Weights.GPUWireF32(); err != nil {
			return false
		}
		if _, err := g.HH.Weights.GPUWireF32(); err != nil {
			return false
		}
	}
	return true
}

func packGateF32(g *Gate, hid, in int) ([]float32, error) {
	ihW, err := g.IH.Weights.GPUWireF32()
	if err != nil {
		return nil, err
	}
	hhW, err := g.HH.Weights.GPUWireF32()
	if err != nil {
		return nil, err
	}
	ihN, hhN := hid*in, hid*hid
	out := make([]float32, ihN+hhN+hid)
	copy(out[:ihN], ihW)
	copy(out[ihN:ihN+hhN], hhW)
	for i := 0; i < hid; i++ {
		out[ihN+hhN+i] = float32(g.IH.Weights.Bias[i])
	}
	return out, nil
}

func packLSTMWeightsF32(l *Layer) ([]float32, error) {
	h, in := l.Cfg.HiddenSize, l.Cfg.InputSize
	gateN := l.Cfg.GateSize()
	out := make([]float32, l.Cfg.WeightCount())
	for i, g := range l.gates() {
		pack, err := packGateF32(g, h, in)
		if err != nil {
			return nil, fmt.Errorf("lstm gate %d: %w", i, err)
		}
		copy(out[i*gateN:(i+1)*gateN], pack)
	}
	return out, nil
}

// ForwardWebGPU — fused LSTM recurrence on device (FormatNone f32).
func ForwardWebGPU[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("lstm: BackendWebGPU but no device (no host fake)")
	}
	if !gpuLSTMOK(l) {
		return forwardViaDense(l, input)
	}
	lay, err := parseLayout(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	weights, err := packLSTMWeightsF32(l)
	if err != nil {
		return nil, nil, err
	}

	pre = core.NewTensor[T](lay.batch, lay.seq, 5*lay.hid)
	post = core.NewTensor[T](lay.batch, lay.seq, lay.hid)
	inF := core.SliceAsFloat32(input.Data)
	preF := make([]float32, lay.batch*lay.seq*5*lay.hid)
	postF := make([]float32, lay.batch*lay.seq*lay.hid)
	if err := webgpu.LSTMForwardSeq(inF, weights, preF, postF, lay.batch, lay.seq, lay.in, lay.hid); err != nil {
		return nil, nil, err
	}
	core.SliceFromFloat32(preF, pre.Data)
	core.SliceFromFloat32(postF, post.Data)
	return pre, post, nil
}

// BackwardWebGPU — BPTT with on-device step DX/DW (FormatNone f32).
func BackwardWebGPU[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("lstm: BackendWebGPU but no device (no host fake)")
	}
	if !gpuLSTMOK(l) {
		return backwardViaDense(l, gradOut, input, pre)
	}
	lay, err := parseLayout(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	if gradOut == nil || pre == nil {
		return nil, nil, fmt.Errorf("lstm: nil gradOut/pre")
	}
	weights, err := packLSTMWeightsF32(l)
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

	gradOutF := core.SliceAsFloat32(gradOut.Data)
	inF := core.SliceAsFloat32(input.Data)
	preF := core.SliceAsFloat32(pre.Data)
	gradInF := make([]float32, lay.batch*lay.seq*lay.in)
	gradWF := make([]float32, len(weights))
	if err := webgpu.LSTMBackwardSeq(gradOutF, inF, preF, weights, gradInF, gradWF, lay.batch, lay.seq, lay.in, lay.hid, tileSize); err != nil {
		return nil, nil, err
	}

	gradIn = core.NewTensor[T](lay.batch, lay.seq, lay.in)
	core.SliceFromFloat32(gradInF, gradIn.Data)
	gradW = core.NewTensor[T](1, l.GradWSize())
	core.SliceFromFloat32(gradWF, gradW.Data)
	return gradIn, gradW, nil
}
