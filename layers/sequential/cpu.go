package sequential

import (
	"fmt"

	"github.com/openfluke/welvet/core"
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
	ops := l.ChildOps()
	if len(ops) == 0 {
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
	for i, ch := range ops {
		p, o, err := callFwd(ch, current)
		if err != nil {
			return nil, nil, fmt.Errorf("sequential fwd child %d: %w", i, err)
		}
		lastPre = p
		current = o
	}
	pre = unflatten(lastPre, lay, l.Cfg.Dim)
	post = unflatten(current, lay, l.Cfg.Dim)
	return pre, post, nil
}

func backwardHost[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	_ = pre
	ops := l.ChildOps()
	if len(ops) == 0 {
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
	n := len(ops)
	ins := make([]*core.Tensor[T], n)
	pres := make([]*core.Tensor[T], n)
	current := flatten(input, lay)
	for i, ch := range ops {
		ins[i] = current
		p, o, err := callFwd(ch, current)
		if err != nil {
			return nil, nil, fmt.Errorf("sequential recompute child %d: %w", i, err)
		}
		pres[i] = p
		current = o
	}
	gy := flatten(gradOut, lay)
	dWs := make([]*core.Tensor[T], n)
	for i := n - 1; i >= 0; i-- {
		gx, dw, err := callBwd(ops[i], gy, ins[i], pres[i])
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
		n := childGradWSize(ops[i])
		if n == 0 {
			continue
		}
		if dw == nil || dw.Len() < n {
			return nil, nil, fmt.Errorf("sequential: nil/short dW child %d", i)
		}
		copy(gradW.Data[off:off+n], dw.Data[:n])
		off += n
	}
	return gradIn, gradW, nil
}
