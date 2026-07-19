package convt1

import (
	"fmt"

	"github.com/openfluke/welvet/core"
)

// ForwardCPUTiled — loom scatter + activation.
func ForwardCPUTiled[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	return forwardHost(l, input)
}

// BackwardCPUTiled — activation Jacobian + loom adjoint.
func BackwardCPUTiled[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	return backwardHost(l, gradOut, input, pre)
}

func forwardHost[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	lay, err := parseLayout(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	w, err := l.Proj.Weights.FlattenF32()
	if err != nil {
		return nil, nil, fmt.Errorf("convt1 weights: %w", err)
	}
	wantW := lay.filters * lay.inC * lay.kSize
	if len(w) < wantW {
		return nil, nil, fmt.Errorf("convt1: weight short %d < %d", len(w), wantW)
	}
	acc := make([]float64, lay.batch*lay.filters*lay.outLen)
	for b := 0; b < lay.batch; b++ {
		for ic := 0; ic < lay.inC; ic++ {
			for iw := 0; iw < lay.seq; iw++ {
				inputVal := core.AsFloat64(input.Data[b*lay.inC*lay.seq+ic*lay.seq+iw])
				for f := 0; f < lay.filters; f++ {
					for k := 0; k < lay.kSize; k++ {
						ow := iw*lay.stride - lay.padding + k
						if ow < 0 || ow >= lay.outLen {
							continue
						}
						outIdx := b*lay.filters*lay.outLen + f*lay.outLen + ow
						kWIdx := f*lay.inC*lay.kSize + ic*lay.kSize + k
						acc[outIdx] += inputVal * float64(w[kWIdx])
					}
				}
			}
		}
	}
	pre = core.NewTensor[T](lay.batch, lay.filters, lay.outLen)
	post = core.NewTensor[T](lay.batch, lay.filters, lay.outLen)
	act := l.Cfg.Activation
	for i := range acc {
		pre.Data[i] = core.FromFloat64[T](acc[i])
		post.Data[i] = core.Activate(pre.Data[i], act)
	}
	return pre, post, nil
}

func backwardHost[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	lay, err := parseLayout(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	if gradOut == nil || pre == nil {
		return nil, nil, fmt.Errorf("convt1: nil gradOut/pre")
	}
	wantOut := lay.batch * lay.filters * lay.outLen
	if gradOut.Len() < wantOut || pre.Len() < wantOut {
		return nil, nil, fmt.Errorf("convt1: grad/pre short")
	}
	w, err := l.Proj.Weights.FlattenF32()
	if err != nil {
		return nil, nil, fmt.Errorf("convt1 weights: %w", err)
	}
	act := l.Cfg.Activation
	dPre := make([]float64, wantOut)
	for i := 0; i < wantOut; i++ {
		dPre[i] = core.AsFloat64(gradOut.Data[i]) * core.AsFloat64(core.ActivateDeriv(pre.Data[i], act))
	}
	gInAcc := make([]float64, lay.batch*lay.inC*lay.seq)
	gWAcc := make([]float64, lay.filters*lay.inC*lay.kSize)
	for b := 0; b < lay.batch; b++ {
		for ic := 0; ic < lay.inC; ic++ {
			for iw := 0; iw < lay.seq; iw++ {
				xin := core.AsFloat64(input.Data[b*lay.inC*lay.seq+ic*lay.seq+iw])
				for f := 0; f < lay.filters; f++ {
					for k := 0; k < lay.kSize; k++ {
						ow := iw*lay.stride - lay.padding + k
						if ow < 0 || ow >= lay.outLen {
							continue
						}
						outIdx := b*lay.filters*lay.outLen + f*lay.outLen + ow
						kWIdx := f*lay.inC*lay.kSize + ic*lay.kSize + k
						gp := dPre[outIdx]
						gWAcc[kWIdx] += xin * gp
						gInAcc[b*lay.inC*lay.seq+ic*lay.seq+iw] += float64(w[kWIdx]) * gp
					}
				}
			}
		}
	}
	gradIn = core.NewTensor[T](lay.batch, lay.inC, lay.seq)
	for i := range gradIn.Data {
		gradIn.Data[i] = core.FromFloat64[T](gInAcc[i])
	}
	gradW = core.NewTensor[T](lay.filters, lay.inC*lay.kSize)
	for i := range gWAcc {
		gradW.Data[i] = core.FromFloat64[T](gWAcc[i])
	}
	return gradIn, gradW, nil
}
