package cnn3

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
)

// ForwardCPUTiled — im2col + Dense CPU MatVec.
func ForwardCPUTiled[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	return forwardViaDense(l, input)
}

// BackwardCPUTiled — Dense bwd + col2im.
func BackwardCPUTiled[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	return backwardViaDense(l, gradOut, input, pre)
}

func forwardViaDense[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	lay, err := parseLayout(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	col := im2col(input, lay)
	preFlat, postFlat, err := dense.Forward(l.Proj, col)
	if err != nil {
		return nil, nil, fmt.Errorf("cnn3 dense fwd: %w", err)
	}
	pre = loomFromDense(preFlat.Data, lay.batch, lay.filters, lay.outD, lay.outH, lay.outW)
	post = loomFromDense(postFlat.Data, lay.batch, lay.filters, lay.outD, lay.outH, lay.outW)
	return pre, post, nil
}

func backwardViaDense[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	lay, err := parseLayout(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	if gradOut == nil || pre == nil {
		return nil, nil, fmt.Errorf("cnn3: nil gradOut/pre")
	}
	wantOut := lay.batch * lay.filters * lay.outD * lay.outH * lay.outW
	if gradOut.Len() < wantOut || pre.Len() < wantOut {
		return nil, nil, fmt.Errorf("cnn3: grad/pre short")
	}
	col := im2col(input, lay)
	gy := denseFromLoom(gradOut.Data, lay.batch, lay.filters, lay.outD, lay.outH, lay.outW)
	preFlat := denseFromLoom(pre.Data, lay.batch, lay.filters, lay.outD, lay.outH, lay.outW)
	gxCol, dW, err := dense.Backward(l.Proj, gy, col, preFlat)
	if err != nil {
		return nil, nil, fmt.Errorf("cnn3 dense bwd: %w", err)
	}
	gradIn = col2im(gxCol, lay)
	return gradIn, dW, nil
}
