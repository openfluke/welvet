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

func forwardHost[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	lay, err := parseLayout(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	emb := lay.emb
	shape := outShape(lay)
	pre = core.NewTensor[T](shape...)
	n := lay.nTok * emb

	// Fast path: gather float32 rows without converting the whole vocab table (Lucy-class).
	if w, ok := l.Weights.MasterF32(); ok {
		if out, ok := any(pre.Data).([]float32); ok {
			for t := 0; t < lay.nTok; t++ {
				id := tokenID(input, lay, t)
				base := id * emb
				if base+emb > len(w) {
					return nil, nil, fmt.Errorf("embedding: token %d OOB", id)
				}
				copy(out[t*emb:(t+1)*emb], w[base:base+emb])
			}
			if l.Core.Activation != core.ActivationLinear {
				for i := 0; i < n; i++ {
					pre.Data[i] = core.Activate(pre.Data[i], l.Core.Activation)
				}
			}
			post = pre
			return pre, post, nil
		}
		for t := 0; t < lay.nTok; t++ {
			id := tokenID(input, lay, t)
			base := id * emb
			if base+emb > len(w) {
				return nil, nil, fmt.Errorf("embedding: token %d OOB", id)
			}
			ob := t * emb
			for j := 0; j < emb; j++ {
				pre.Data[ob+j] = core.FromFloat64[T](float64(w[base+j]))
			}
		}
		if l.Core.Activation != core.ActivationLinear {
			for i := 0; i < n; i++ {
				pre.Data[i] = core.Activate(pre.Data[i], l.Core.Activation)
			}
		}
		post = pre
		return pre, post, nil
	}

	table, err := weightTableF64(l)
	if err != nil {
		return nil, nil, fmt.Errorf("embedding wire: %w", err)
	}
	for t := 0; t < lay.nTok; t++ {
		id := tokenID(input, lay, t)
		base := id * emb
		if base+emb > len(table) {
			return nil, nil, fmt.Errorf("embedding: token %d OOB", id)
		}
		ob := t * emb
		for j := 0; j < emb; j++ {
			pre.Data[ob+j] = core.FromFloat64[T](table[base+j])
		}
	}
	if l.Core.Activation != core.ActivationLinear {
		for i := 0; i < n; i++ {
			pre.Data[i] = core.Activate(pre.Data[i], l.Core.Activation)
		}
	}
	post = pre
	return pre, post, nil
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
