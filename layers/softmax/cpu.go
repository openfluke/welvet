package softmax

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/openfluke/welvet/core"
)

// ForwardCPUTiled — stable Softmax on host (all Kind variants).
func ForwardCPUTiled[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	return forwardHost(l, input)
}

// BackwardCPUTiled — Jacobian × 1/T; gradW nil.
func BackwardCPUTiled[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	return backwardHost(l, gradOut, input, pre)
}

func logitsFromRow[T core.Numeric](data []T, start, cols int, invT float64) []float32 {
	logits := make([]float32, cols)
	for i := 0; i < cols; i++ {
		logits[i] = float32(core.AsFloat64(data[start+i]) * invT)
	}
	return logits
}

func applyMask(logits []float32, mask []bool, globalStart int) {
	for i := range logits {
		if globalStart+i < len(mask) && !mask[globalStart+i] {
			logits[i] = -1e9
		}
	}
}

func gumbelNoise(logits []float32) []float32 {
	noisy := make([]float32, len(logits))
	for i, v := range logits {
		u := rand.Float32()
		if u < 1e-10 {
			u = 1e-10
		}
		gumbel := -float32(math.Log(-math.Log(float64(u))))
		noisy[i] = v + gumbel
	}
	return noisy
}

func probsForKind(cfg Config, logits []float32, globalStart int) []float32 {
	switch cfg.Kind {
	case KindGumbel:
		return Softmax(gumbelNoise(logits))
	case KindMasked:
		masked := make([]float32, len(logits))
		copy(masked, logits)
		applyMask(masked, cfg.Mask, globalStart)
		return Softmax(masked)
	case KindSparse:
		return SoftmaxSparseHelper(logits)
	case KindEntmax:
		alpha := float32(cfg.EntmaxAlpha)
		if alpha == 0 {
			alpha = 1.5
		}
		return SoftmaxEntmaxHelper(logits, alpha)
	default:
		return Softmax(logits)
	}
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
		logits := logitsFromRow(input.Data, start, cols, invT)
		probs := probsForKind(l.Cfg, logits, start)
		var sum float64
		for _, p := range probs {
			sum += float64(p)
		}
		if sum == 0 {
			return nil, nil, fmt.Errorf("softmax: zero sum at row %d", r)
		}
		for i := 0; i < cols; i++ {
			out[start+i] = core.FromFloat64[T](float64(probs[i]))
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
		probs := make([]float32, cols)
		grads := make([]float32, cols)
		for i := 0; i < cols; i++ {
			probs[i] = float32(core.AsFloat64(pre.Data[start+i]))
			grads[i] = float32(core.AsFloat64(gradOut.Data[start+i]))
		}
		if l.Cfg.Kind == KindMasked {
			for i := 0; i < cols; i++ {
				if start+i < len(l.Cfg.Mask) && !l.Cfg.Mask[start+i] {
					grads[i] = 0
				}
			}
		}
		gradLogits := SoftmaxBackward(grads, probs)
		for i := 0; i < cols; i++ {
			dx[start+i] = core.FromFloat64[T](float64(gradLogits[i])*invT)
		}
	}
	gradIn = core.NewTensor[T](lay.shape...)
	copy(gradIn.Data, dx)
	return gradIn, nil, nil
}
