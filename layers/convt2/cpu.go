package convt2

import (
	"fmt"

	"github.com/openfluke/welvet/core"
)

func ForwardCPUTiled[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	return forwardHost(l, input)
}

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
		return nil, nil, fmt.Errorf("convt2 weights: %w", err)
	}
	kk := lay.kSize * lay.kSize
	wantW := lay.filters * lay.inC * kk
	if len(w) < wantW {
		return nil, nil, fmt.Errorf("convt2: weight short %d < %d", len(w), wantW)
	}
	spOut := spatialOut(lay)
	spIn := spatialIn(lay)
	acc := make([]float64, lay.batch*lay.filters*spOut)
	for b := 0; b < lay.batch; b++ {
		for ic := 0; ic < lay.inC; ic++ {
			for ih := 0; ih < lay.inH; ih++ {
				for iw := 0; iw < lay.inW; iw++ {
					inputVal := core.AsFloat64(input.Data[b*lay.inC*spIn+ic*spIn+ih*lay.inW+iw])
					for f := 0; f < lay.filters; f++ {
						for kh := 0; kh < lay.kSize; kh++ {
							for kw := 0; kw < lay.kSize; kw++ {
								oh := ih*lay.stride - lay.padding + kh
								ow := iw*lay.stride - lay.padding + kw
								if oh < 0 || oh >= lay.outH || ow < 0 || ow >= lay.outW {
									continue
								}
								outIdx := b*lay.filters*spOut + f*spOut + oh*lay.outW + ow
								kWIdx := f*lay.inC*kk + ic*kk + kh*lay.kSize + kw
								acc[outIdx] += inputVal * float64(w[kWIdx])
							}
						}
					}
				}
			}
		}
	}
	pre = core.NewTensor[T](lay.batch, lay.filters, lay.outH, lay.outW)
	post = core.NewTensor[T](lay.batch, lay.filters, lay.outH, lay.outW)
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
		return nil, nil, fmt.Errorf("convt2: nil gradOut/pre")
	}
	spOut := spatialOut(lay)
	spIn := spatialIn(lay)
	wantOut := lay.batch * lay.filters * spOut
	if gradOut.Len() < wantOut || pre.Len() < wantOut {
		return nil, nil, fmt.Errorf("convt2: grad/pre short")
	}
	w, err := l.Proj.Weights.FlattenF32()
	if err != nil {
		return nil, nil, fmt.Errorf("convt2 weights: %w", err)
	}
	kk := lay.kSize * lay.kSize
	act := l.Cfg.Activation
	dPre := make([]float64, wantOut)
	for i := 0; i < wantOut; i++ {
		dPre[i] = core.AsFloat64(gradOut.Data[i]) * core.AsFloat64(core.ActivateDeriv(pre.Data[i], act))
	}
	gInAcc := make([]float64, lay.batch*lay.inC*spIn)
	gWAcc := make([]float64, lay.filters*lay.inC*kk)
	for b := 0; b < lay.batch; b++ {
		for ic := 0; ic < lay.inC; ic++ {
			for ih := 0; ih < lay.inH; ih++ {
				for iw := 0; iw < lay.inW; iw++ {
					xin := core.AsFloat64(input.Data[b*lay.inC*spIn+ic*spIn+ih*lay.inW+iw])
					for f := 0; f < lay.filters; f++ {
						for kh := 0; kh < lay.kSize; kh++ {
							for kw := 0; kw < lay.kSize; kw++ {
								oh := ih*lay.stride - lay.padding + kh
								ow := iw*lay.stride - lay.padding + kw
								if oh < 0 || oh >= lay.outH || ow < 0 || ow >= lay.outW {
									continue
								}
								outIdx := b*lay.filters*spOut + f*spOut + oh*lay.outW + ow
								kWIdx := f*lay.inC*kk + ic*kk + kh*lay.kSize + kw
								gp := dPre[outIdx]
								gWAcc[kWIdx] += xin * gp
								gInAcc[b*lay.inC*spIn+ic*spIn+ih*lay.inW+iw] += float64(w[kWIdx]) * gp
							}
						}
					}
				}
			}
		}
	}
	gradIn = core.NewTensor[T](lay.batch, lay.inC, lay.inH, lay.inW)
	for i := range gradIn.Data {
		gradIn.Data[i] = core.FromFloat64[T](gInAcc[i])
	}
	gradW = core.NewTensor[T](lay.filters, lay.inC*kk)
	for i := range gWAcc {
		gradW.Data[i] = core.FromFloat64[T](gWAcc[i])
	}
	return gradIn, gradW, nil
}
