package embedding

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
)

// ForwardCPUTiled — host gather from weight wire.
func ForwardCPUTiled[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	return forwardHost(l, input)
}

// BackwardCPUTiled — host scatter into dW; gradIn zeros.
func BackwardCPUTiled[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	return backwardHost(l, gradOut, input, pre)
}

func weightTableF64(l *Layer) ([]float64, error) {
	// FormatNone+Float32: read master live (finite-diff / SGD edits masterF32).
	if l.Weights.Format == quant.FormatNone && l.Weights.DType == core.DTypeFloat32 {
		if w, ok := l.Weights.MasterF32(); ok {
			out := make([]float64, len(w))
			for i, v := range w {
				out[i] = float64(v)
			}
			return out, nil
		}
	}
	return l.Weights.WireF64()
}

func forwardHost[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	lay, err := parseLayout(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	table, err := weightTableF64(l)
	if err != nil {
		return nil, nil, fmt.Errorf("embedding wire: %w", err)
	}
	emb := lay.emb
	out := make([]T, lay.nTok*emb)
	row := make([]float64, emb)
	for t := 0; t < lay.nTok; t++ {
		id := tokenID(input, lay, t)
		base := id * emb
		if base+emb > len(table) {
			return nil, nil, fmt.Errorf("embedding: token %d OOB", id)
		}
		copy(row, table[base:base+emb])
		ob := t * emb
		for j := 0; j < emb; j++ {
			out[ob+j] = core.FromFloat64[T](row[j])
		}
	}
	// Apply activation (usually Linear).
	act := l.Core.Activation
	for i := range out {
		out[i] = core.Activate(out[i], act)
	}
	shape := outShape(lay)
	pre = core.NewTensor[T](shape...)
	post = core.NewTensor[T](shape...)
	copy(pre.Data, out)
	copy(post.Data, out)
	return pre, post, nil
}

func backwardHost[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	_ = pre
	lay, err := parseLayout(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	if gradOut == nil || gradOut.Len() < lay.nTok*lay.emb {
		return nil, nil, fmt.Errorf("embedding: gradOut short")
	}
	emb := lay.emb
	dW := make([]T, lay.vocab*emb)
	for t := 0; t < lay.nTok; t++ {
		id := tokenID(input, lay, t)
		ob := t * emb
		rb := id * emb
		for j := 0; j < emb; j++ {
			dW[rb+j] = core.FromFloat64[T](core.AsFloat64(dW[rb+j]) + core.AsFloat64(gradOut.Data[ob+j]))
		}
	}
	gradW = core.NewTensor[T](lay.vocab, emb)
	copy(gradW.Data, dW)
	gradIn = core.NewTensor[T](input.Shape...)
	// zeros — no gradient through discrete token IDs
	return gradIn, gradW, nil
}
