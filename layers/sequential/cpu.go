package sequential

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
)

// ForwardCPUTiled — chain Dense children.
func ForwardCPUTiled[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	return forwardHost(l, input)
}

// BackwardCPUTiled — reverse Dense chain; gradW concat.
func BackwardCPUTiled[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	return backwardHost(l, gradOut, input, pre)
}

func forwardHost[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if len(l.Children) == 0 {
		out := core.NewTensor[T](input.Shape...)
		copy(out.Data, input.Data)
		return out, out, nil
	}
	lay, err := parseLayout(l.Cfg.Dim, input)
	if err != nil {
		return nil, nil, err
	}
	current := flatten(input, lay)
	var lastPre *core.Tensor[T]
	for i, ch := range l.Children {
		p, o, err := dense.Forward(ch, current)
		if err != nil {
			return nil, nil, fmt.Errorf("sequential fwd child %d: %w", i, err)
		}
		lastPre = p
		current = o
	}
	// pre = first child's pre (or last); post = final activation, unflattened
	pre = unflatten(lastPre, lay, l.Cfg.Dim)
	post = unflatten(current, lay, l.Cfg.Dim)
	return pre, post, nil
}

func backwardHost[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	_ = pre
	if len(l.Children) == 0 {
		gi := core.NewTensor[T](input.Shape...)
		if gradOut != nil {
			copy(gi.Data, gradOut.Data)
		}
		return gi, nil, nil
	}
	lay, err := parseLayout(l.Cfg.Dim, input)
	if err != nil {
		return nil, nil, err
	}
	// Recompute forward intermediates.
	n := len(l.Children)
	ins := make([]*core.Tensor[T], n)
	pres := make([]*core.Tensor[T], n)
	current := flatten(input, lay)
	for i, ch := range l.Children {
		ins[i] = current
		p, o, err := dense.Forward(ch, current)
		if err != nil {
			return nil, nil, fmt.Errorf("sequential recompute child %d: %w", i, err)
		}
		pres[i] = p
		current = o
	}
	gy := flatten(gradOut, lay)
	dWs := make([]*core.Tensor[T], n)
	for i := n - 1; i >= 0; i-- {
		gx, dw, err := dense.Backward(l.Children[i], gy, ins[i], pres[i])
		if err != nil {
			return nil, nil, fmt.Errorf("sequential bwd child %d: %w", i, err)
		}
		dWs[i] = dw
		gy = gx
	}
	gradIn = unflatten(gy, lay, l.Cfg.Dim)
	need := l.GradWSize()
	gradW = core.NewTensor[T](need)
	off := 0
	for i, dw := range dWs {
		if dw == nil {
			return nil, nil, fmt.Errorf("sequential: nil dW child %d", i)
		}
		copy(gradW.Data[off:], dw.Data)
		off += dw.Len()
	}
	return gradIn, gradW, nil
}
