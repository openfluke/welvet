package metacognition

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
)

func ForwardCPUTiled[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	return forwardHost(l, input)
}

func BackwardCPUTiled[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	return backwardHost(l, gradOut, input, pre)
}

func flatten[T core.Numeric](cfg Config, input *core.Tensor[T]) (*core.Tensor[T], error) {
	dim := cfg.Dim
	switch {
	case len(input.Shape) == 2 && input.Shape[1] == dim:
		return input, nil
	case len(input.Shape) == 3 && input.Shape[2] == dim:
		n := input.Shape[0] * input.Shape[1]
		out := core.NewTensor[T](n, dim)
		copy(out.Data, input.Data)
		return out, nil
	default:
		return nil, fmt.Errorf("metacognition: shape %v dim %d", input.Shape, dim)
	}
}

func unflatten[T core.Numeric](cfg Config, flat *core.Tensor[T], shape []int) *core.Tensor[T] {
	out := core.NewTensor[T](shape...)
	copy(out.Data, flat.Data)
	return out
}

func forwardHost[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	flat, err := flatten(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	inStats := computeStats(flat)
	l.lastGate = 1

	preF, postF, err := dense.Forward(l.Observed, flat)
	if err != nil {
		return nil, nil, err
	}
	outStats := computeStats(postF)

	effect := EffectNone
	for i := range l.Rules {
		r := &l.Rules[i]
		if r.Cooldown > 0 && r.fireCount < r.Cooldown {
			r.fireCount++
			continue
		}
		if evalRule(r, inStats, outStats) {
			effect = r.Effect
			r.fireCount = 0
		} else {
			r.fireCount++
		}
	}

	switch effect {
	case EffectIdentity:
		if err := resetIdentity(l.Observed); err != nil {
			return nil, nil, err
		}
		preF, postF, err = dense.Forward(l.Observed, flat)
		if err != nil {
			return nil, nil, err
		}
	case EffectGate:
		l.lastGate = 0.5
		for i := range postF.Data {
			postF.Data[i] = core.FromFloat64[T](core.AsFloat64(postF.Data[i]) * l.lastGate)
		}
	case EffectPassthrough:
		preF = core.NewTensor[T](flat.Shape...)
		postF = core.NewTensor[T](flat.Shape...)
		copy(preF.Data, flat.Data)
		copy(postF.Data, flat.Data)
	}

	pre = unflatten(l.Cfg, preF, input.Shape)
	post = unflatten(l.Cfg, postF, input.Shape)
	return pre, post, nil
}

func backwardHost[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	flat, err := flatten(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	gy, err := flatten(l.Cfg, gradOut)
	if err != nil {
		return nil, nil, err
	}
	if l.lastGate != 1 && l.lastGate != 0 {
		for i := range gy.Data {
			gy.Data[i] = core.FromFloat64[T](core.AsFloat64(gy.Data[i]) * l.lastGate)
		}
	}
	preF, err := flatten(l.Cfg, pre)
	if err != nil {
		// recompute
		p, _, e := dense.Forward(l.Observed, flat)
		if e != nil {
			return nil, nil, e
		}
		preF = p
	}
	gx, dw, err := dense.Backward(l.Observed, gy, flat, preF)
	if err != nil {
		return nil, nil, err
	}
	gradIn = unflatten(l.Cfg, gx, input.Shape)
	return gradIn, dw, nil
}

func resetIdentity(obs *dense.Layer) error {
	if obs == nil || obs.Weights == nil {
		return fmt.Errorf("metacognition: nil observed")
	}
	rows, cols := obs.Weights.Rows, obs.Weights.Cols
	if rows != cols {
		return nil // only square
	}
	w := make([]float32, rows*cols)
	for i := 0; i < rows; i++ {
		w[i*cols+i] = 1
	}
	return obs.Weights.SetFromF32(w)
}
