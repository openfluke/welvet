package residual

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
)

// ForwardCPUTiled — y = F(x) + x.
func ForwardCPUTiled[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	return forwardHost(l, input)
}

// BackwardCPUTiled — ∂F/∂x + skip grad; gradW concat.
func BackwardCPUTiled[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	return backwardHost(l, gradOut, input, pre)
}

func forwardHost[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	lay, err := parseLayout(l.Cfg.Dim, input)
	if err != nil {
		return nil, nil, err
	}
	xFlat := flatten(input, lay)
	fx := xFlat
	var lastPre *core.Tensor[T]
	if len(l.Children) == 0 {
		// identity F → y = 2x; keep classic residual as F=0 → y=x via empty add of zeros?
		// Empty F: treat as F(x)=0 so y=x (loom identity when no transform).
		out := core.NewTensor[T](input.Shape...)
		copy(out.Data, input.Data)
		return out, out, nil
	}
	current := xFlat
	for i, ch := range l.Children {
		p, o, err := dense.Forward(ch, current)
		if err != nil {
			return nil, nil, fmt.Errorf("residual F fwd child %d: %w", i, err)
		}
		lastPre = p
		current = o
	}
	fx = current
	// y = fx + x
	y := core.NewTensor[T](fx.Shape...)
	for i := range y.Data {
		y.Data[i] = core.FromFloat64[T](core.AsFloat64(fx.Data[i]) + core.AsFloat64(xFlat.Data[i]))
	}
	pre = unflatten(lastPre, lay, l.Cfg.Dim)
	post = unflatten(y, lay, l.Cfg.Dim)
	return pre, post, nil
}

func backwardHost[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	_ = pre
	lay, err := parseLayout(l.Cfg.Dim, input)
	if err != nil {
		return nil, nil, err
	}
	gy := flatten(gradOut, lay)
	xFlat := flatten(input, lay)

	if len(l.Children) == 0 {
		// y=x → gradIn = gy
		return unflatten(gy, lay, l.Cfg.Dim), nil, nil
	}

	n := len(l.Children)
	ins := make([]*core.Tensor[T], n)
	pres := make([]*core.Tensor[T], n)
	current := xFlat
	for i, ch := range l.Children {
		ins[i] = current
		p, o, err := dense.Forward(ch, current)
		if err != nil {
			return nil, nil, fmt.Errorf("residual F recompute child %d: %w", i, err)
		}
		pres[i] = p
		current = o
	}

	// ∂L/∂fx = gy; then reverse through F
	dWs := make([]*core.Tensor[T], n)
	gFx := gy
	for i := n - 1; i >= 0; i-- {
		gx, dw, err := dense.Backward(l.Children[i], gFx, ins[i], pres[i])
		if err != nil {
			return nil, nil, fmt.Errorf("residual F bwd child %d: %w", i, err)
		}
		dWs[i] = dw
		gFx = gx
	}
	// gradIn = ∂F/∂x + ∂L/∂y (skip)
	gin := core.NewTensor[T](gFx.Shape...)
	for i := range gin.Data {
		gin.Data[i] = core.FromFloat64[T](core.AsFloat64(gFx.Data[i]) + core.AsFloat64(gy.Data[i]))
	}
	gradIn = unflatten(gin, lay, l.Cfg.Dim)
	need := l.GradWSize()
	gradW = core.NewTensor[T](need)
	off := 0
	for i, dw := range dWs {
		if dw == nil {
			return nil, nil, fmt.Errorf("residual: nil dW child %d", i)
		}
		copy(gradW.Data[off:], dw.Data)
		off += dw.Len()
	}
	return gradIn, gradW, nil
}
