package softmax

import (
	"fmt"
	"math"

	"github.com/openfluke/welvet/core"
)

// ForwardCPUTiled — stable Softmax on host.
func ForwardCPUTiled[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	return forwardHost(l, input)
}

// BackwardCPUTiled — Jacobian × 1/T; gradW nil.
func BackwardCPUTiled[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	return backwardHost(l, gradOut, input, pre)
}

func forwardHost[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	lay, err := parseLayout(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	out := make([]T, lay.n)
	invT := 1.0 / lay.temp
	for r := 0; r < lay.rows; r++ {
		start := r * lay.cols
		end := start + lay.cols
		if end > lay.n {
			end = lay.n
		}
		cols := end - start
		// max of x/T
		maxZ := core.AsFloat64(input.Data[start]) * invT
		for i := 1; i < cols; i++ {
			z := core.AsFloat64(input.Data[start+i]) * invT
			if z > maxZ {
				maxZ = z
			}
		}
		var sum float64
		exps := make([]float64, cols)
		for i := 0; i < cols; i++ {
			z := core.AsFloat64(input.Data[start+i])*invT - maxZ
			e := math.Exp(z)
			exps[i] = e
			sum += e
		}
		if sum == 0 {
			return nil, nil, fmt.Errorf("softmax: zero sum at row %d", r)
		}
		for i := 0; i < cols; i++ {
			out[start+i] = core.FromFloat64[T](exps[i] / sum)
		}
	}
	pre = core.NewTensor[T](lay.shape...)
	post = core.NewTensor[T](lay.shape...)
	copy(pre.Data, out)
	copy(post.Data, out)
	return pre, post, nil
}

func backwardHost[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	_ = input
	lay, err := parseLayout(l.Cfg, pre)
	if err != nil {
		// fall back: parse from gradOut shape via a fake using cfg on gradOut
		lay, err = parseLayout(l.Cfg, gradOut)
		if err != nil {
			return nil, nil, err
		}
	}
	if gradOut == nil || pre == nil || gradOut.Len() < lay.n || pre.Len() < lay.n {
		return nil, nil, fmt.Errorf("softmax: nil/short gradOut/pre")
	}
	invT := 1.0 / lay.temp
	dx := make([]T, lay.n)
	for r := 0; r < lay.rows; r++ {
		start := r * lay.cols
		end := start + lay.cols
		if end > lay.n {
			end = lay.n
		}
		cols := end - start
		var dot float64
		for i := 0; i < cols; i++ {
			dot += core.AsFloat64(gradOut.Data[start+i]) * core.AsFloat64(pre.Data[start+i])
		}
		for i := 0; i < cols; i++ {
			y := core.AsFloat64(pre.Data[start+i])
			g := core.AsFloat64(gradOut.Data[start+i])
			dx[start+i] = core.FromFloat64[T](y * (g - dot) * invT)
		}
	}
	gradIn = core.NewTensor[T](lay.shape...)
	copy(gradIn.Data, dx)
	return gradIn, nil, nil
}
