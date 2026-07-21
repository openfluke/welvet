package cnn1

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/webgpu"
)

// ForwardWebGPU — FormatNone f32: on-device tiled conv; else host im2col + Dense WebGPU GEMV.
func ForwardWebGPU[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("cnn1: BackendWebGPU but no device (no host fake)")
	}
	if tiledF32OK(l) {
		return forwardTiledWebGPU(l, input)
	}
	// Quant / other FormatNone dtypes: host im2col reshuffle, then real on-device Dense matmul.
	return forwardViaDense(l, input)
}

// BackwardWebGPU — reverse of ForwardWebGPU dispatch.
func BackwardWebGPU[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if !webgpu.Available() {
		return nil, nil, fmt.Errorf("cnn1: BackendWebGPU but no device (no host fake)")
	}
	if tiledF32OK(l) && webgpu.CNNTiledBwdOK(l.Cfg.Activation) {
		return backwardTiledWebGPU(l, gradOut, input, pre)
	}
	return backwardViaDense(l, gradOut, input, pre)
}

func tiledF32OK(l *Layer) bool {
	if l == nil || l.Proj == nil || l.Proj.Weights == nil {
		return false
	}
	w := l.Proj.Weights
	return w.Format == quant.FormatNone && w.DType == core.DTypeFloat32
}

func forwardTiledWebGPU[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	lay, err := parseLayout(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	pre = core.NewTensor[T](lay.batch, lay.filters, lay.outLen)
	post = core.NewTensor[T](lay.batch, lay.filters, lay.outLen)

	xF := core.SliceAsFloat32(input.Data[:lay.batch*lay.inC*lay.seq])
	preF := make([]float32, lay.batch*lay.filters*lay.outLen)
	postF := make([]float32, len(preF))

	wF, err := l.Proj.Weights.GPUWireF32()
	if err != nil {
		return nil, nil, fmt.Errorf("cnn1 tiled fwd weights: %w", err)
	}
	cfg := webgpu.CNN1Config{
		Batch: lay.batch, InC: lay.inC, InL: lay.seq,
		OutC: lay.filters, OutL: lay.outLen,
		KSize: lay.kSize, Stride: lay.stride, Padding: lay.padding,
		MultiCore: l.Exec.MultiCore || l.Core.MultiCore,
	}
	if err := webgpu.CNN1Forward(xF, wF, preF, cfg); err != nil {
		return nil, nil, fmt.Errorf("cnn1 tiled fwd: %w", err)
	}
	applyBiasActLoom(preF, postF, l.Proj.Weights.Bias, l.Cfg.Activation, lay.batch, lay.filters, lay.outLen)
	core.SliceFromFloat32(preF, pre.Data)
	core.SliceFromFloat32(postF, post.Data)
	return pre, post, nil
}

func backwardTiledWebGPU[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	lay, err := parseLayout(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	wantOut := lay.batch * lay.filters * lay.outLen
	if gradOut == nil || pre == nil || gradOut.Len() < wantOut || pre.Len() < wantOut {
		return nil, nil, fmt.Errorf("cnn1: grad/pre shape")
	}

	gradIn = core.NewTensor[T](lay.batch, lay.inC, lay.seq)
	gradW = core.NewTensor[T](lay.filters, lay.inC*lay.kSize)

	gyF := core.SliceAsFloat32(gradOut.Data[:wantOut])
	inF := core.SliceAsFloat32(input.Data[:lay.batch*lay.inC*lay.seq])
	preF := core.SliceAsFloat32(pre.Data[:wantOut])
	gxF := make([]float32, lay.batch*lay.inC*lay.seq)
	gwF := make([]float32, lay.filters*lay.inC*lay.kSize)

	wF, err := l.Proj.Weights.GPUWireF32()
	if err != nil {
		return nil, nil, fmt.Errorf("cnn1 tiled bwd weights: %w", err)
	}
	cfg := webgpu.CNN1Config{
		Batch: lay.batch, InC: lay.inC, InL: lay.seq,
		OutC: lay.filters, OutL: lay.outLen,
		KSize: lay.kSize, Stride: lay.stride, Padding: lay.padding,
		MultiCore: l.Exec.MultiCore || l.Core.MultiCore,
	}
	if err := webgpu.CNN1Backward(gyF, wF, inF, preF, gxF, gwF, cfg, l.Cfg.Activation); err != nil {
		return nil, nil, fmt.Errorf("cnn1 tiled bwd: %w", err)
	}
	core.SliceFromFloat32(gxF, gradIn.Data)
	core.SliceFromFloat32(gwF, gradW.Data)
	return gradIn, gradW, nil
}

func applyBiasActLoom(pre, post []float32, bias []float64, act core.ActivationType, batch, filters, spatial int) {
	for b := 0; b < batch; b++ {
		for f := 0; f < filters; f++ {
			for s := 0; s < spatial; s++ {
				i := b*filters*spatial + f*spatial + s
				v := pre[i]
				if bias != nil && f < len(bias) {
					v += float32(bias[f])
					pre[i] = v
				}
				post[i] = float32(core.AsFloat64(core.Activate(v, act)))
			}
		}
	}
}
