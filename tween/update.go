package tween

import (
	"fmt"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/dna"
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/cnn2"
	"github.com/openfluke/welvet/layers/cnn3"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/embedding"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/lstm"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/rnn"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/layers/softmax"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/weights"
)

// applyStoreHebbian updates one weight store from input×gap (Dense out×in layout when shape matches).
func applyStoreHebbian(s *weights.Store, input, gap []float32, layerRate, mom float32, vel *[]float32) error {
	if s == nil || len(gap) == 0 {
		return nil
	}
	w, err := s.FlattenF32()
	if err != nil {
		return err
	}
	outSize, inSize := s.Rows, s.Cols
	need := outSize*inSize + outSize
	if vel == nil {
		tmp := make([]float32, need)
		vel = &tmp
	}
	if len(*vel) != need {
		*vel = make([]float32, need)
	}
	v := *vel

	// Dense-style GEMM outer product when lengths align.
	if len(input) >= inSize && len(gap) >= outSize {
		for out := 0; out < outSize; out++ {
			for in := 0; in < inSize; in++ {
				wIdx := out*inSize + in
				if wIdx >= len(w) {
					continue
				}
				delta := layerRate * input[in%len(input)] * gap[out]
				v[wIdx] = mom*v[wIdx] + (1-mom)*delta
				w[wIdx] += v[wIdx]
			}
			if len(s.Bias) > out {
				bIdx := outSize*inSize + out
				delta := layerRate * gap[out]
				v[bIdx] = mom*v[bIdx] + (1-mom)*delta
				s.Bias[out] += float64(v[bIdx])
			}
		}
	} else {
		// Fallback: scale each weight by mean gap (norms / odd shapes).
		var mean float32
		for _, g := range gap {
			mean += g
		}
		mean /= float32(len(gap))
		for i := range w {
			delta := layerRate * mean * 0.01
			if i < len(v) {
				v[i] = mom*v[i] + (1-mom)*delta
				w[i] += v[i]
			} else {
				w[i] += delta
			}
		}
	}
	return s.SetFromF32(w)
}

func collectDenseChildren(op any) []*dense.Layer {
	switch v := op.(type) {
	case *dense.Layer:
		return []*dense.Layer{v}
	case *mha.Layer:
		return []*dense.Layer{v.Q, v.K, v.V, v.O}
	case *swiglu.Layer:
		return []*dense.Layer{v.Gate, v.Up, v.Down}
	case *cnn1.Layer:
		return []*dense.Layer{v.Proj}
	case *cnn2.Layer:
		return []*dense.Layer{v.Proj}
	case *cnn3.Layer:
		return []*dense.Layer{v.Proj}
	case *rnn.Layer:
		return []*dense.Layer{v.IH, v.HH}
	case *lstm.Layer:
		var out []*dense.Layer
		for _, g := range []*lstm.Gate{v.I, v.F, v.G, v.O} {
			if g != nil {
				out = append(out, g.IH, g.HH)
			}
		}
		return out
	case *sequential.Layer:
		return v.Children
	case *residual.Layer:
		return v.Children
	default:
		return nil
	}
}

func applyGapsLayerwiseAll[T core.Numeric](g *architecture.Grid, s *State[T], lr float32) error {
	order := hopCells(g)
	for i := 0; i < len(order); i++ {
		budget := float32(0)
		if i < len(s.LinkBudgets) {
			budget = s.LinkBudgets[i]
		}
		if budget < 0.2 {
			continue
		}
		layerRate := lr * (0.5 + budget*0.5)
		cell := order[i]
		if cell == nil || cell.Layer.IsDisabled {
			continue
		}
		inputT := s.ForwardActs[i]
		actual := s.ForwardActs[i+1]
		target := s.BackwardTargets[i+1]
		if inputT == nil || actual == nil || target == nil {
			continue
		}
		inF := make([]float32, inputT.Len())
		for j := range inF {
			inF[j] = float32(core.AsFloat64(inputT.Data[j]))
		}
		n := actual.Len()
		if target.Len() < n {
			n = target.Len()
		}
		gap := make([]float32, n)
		for j := 0; j < n; j++ {
			gap[j] = float32(core.AsFloat64(target.Data[j]) - core.AsFloat64(actual.Data[j]))
		}

		children := collectDenseChildren(cell.Op)
		if len(children) > 0 {
			for ci, ch := range children {
				if ch == nil || ch.Weights == nil {
					continue
				}
				key := i*64 + ci
				for len(s.WeightVel) <= key {
					s.WeightVel = append(s.WeightVel, nil)
				}
				if err := applyStoreHebbian(ch.Weights, inF, gap, layerRate, s.Config.Momentum, &s.WeightVel[key]); err != nil {
					return err
				}
			}
			continue
		}

		stores := dna.CollectStores(cell.Op)
		if len(stores) == 0 {
			if _, ok := cell.Op.(*softmax.Layer); ok {
				continue
			}
			continue
		}
		for si, st := range stores {
			key := i*64 + si
			for len(s.WeightVel) <= key {
				s.WeightVel = append(s.WeightVel, nil)
			}
			if err := applyStoreHebbian(st, inF, gap, layerRate, s.Config.Momentum, &s.WeightVel[key]); err != nil {
				return err
			}
		}
	}
	return nil
}

func applyChainRuleAll[T core.Numeric](g *architecture.Grid, s *State[T], lr float32) error {
	order := hopCells(g)
	for i := 0; i < len(order); i++ {
		budget := float32(0)
		if i < len(s.LinkBudgets) {
			budget = s.LinkBudgets[i]
		}
		if budget < 0.2 {
			continue
		}
		layerRate := lr * (0.5 + budget*0.5)
		cell := order[i]
		if cell == nil || cell.Layer.IsDisabled {
			continue
		}
		gW := s.WeightGradients[i]
		if gW == nil {
			continue
		}
		mom := s.Config.Momentum
		if s.WeightVel[i] == nil || len(s.WeightVel[i]) != len(gW.Data) {
			s.WeightVel[i] = make([]float32, len(gW.Data))
		}
		scaled := core.NewTensor[T](gW.Shape...)
		for j := range gW.Data {
			delta := layerRate * gW.Data[j]
			s.WeightVel[i][j] = mom*s.WeightVel[i][j] + (1-mom)*delta
			// ApplyGradSGD does w -= lr * dW; pass negative vel with lr=1.
			scaled.Data[j] = core.FromFloat64[T](-float64(s.WeightVel[i][j]))
		}
		if err := applyGradSGDAny(cell.Op, scaled, 1.0); err != nil {
			return err
		}
	}
	return nil
}

func applyGradSGDAny[T core.Numeric](op any, dW *core.Tensor[T], lr float64) error {
	switch v := op.(type) {
	case *dense.Layer:
		return dense.ApplyGradSGD(v, dW, lr)
	case *mha.Layer:
		return mha.ApplyGradSGD(v, dW, lr)
	case *swiglu.Layer:
		return swiglu.ApplyGradSGD(v, dW, lr)
	case *rmsnorm.Layer:
		return rmsnorm.ApplyGradSGD(v, dW, lr)
	case *layernorm.Layer:
		return layernorm.ApplyGradSGD(v, dW, lr)
	case *cnn1.Layer:
		return cnn1.ApplyGradSGD(v, dW, lr)
	case *cnn2.Layer:
		return cnn2.ApplyGradSGD(v, dW, lr)
	case *cnn3.Layer:
		return cnn3.ApplyGradSGD(v, dW, lr)
	case *rnn.Layer:
		return rnn.ApplyGradSGD(v, dW, lr)
	case *lstm.Layer:
		return lstm.ApplyGradSGD(v, dW, lr)
	case *embedding.Layer:
		return embedding.ApplyGradSGD(v, dW, lr)
	case *sequential.Layer:
		return sequential.ApplyGradSGD(v, dW, lr)
	case *residual.Layer:
		return residual.ApplyGradSGD(v, dW, lr)
	case *softmax.Layer:
		return softmax.ApplyGradSGD(v, dW, lr)
	default:
		return fmt.Errorf("tween: ApplyGradSGD unsupported %T", op)
	}
}

func layerBackwardAny[T core.Numeric](
	cell *architecture.Cell,
	gradOut *core.Tensor[T],
	input, pre *core.Tensor[T],
) (gIn, gW *core.Tensor[T], err error) {
	if cell == nil {
		return nil, nil, fmt.Errorf("nil cell")
	}
	switch cell.Layer.Type {
	case core.LayerDense:
		dl, ok := cell.Op.(*dense.Layer)
		if !ok {
			return nil, nil, fmt.Errorf("dense op %T", cell.Op)
		}
		return dense.Backward(dl, gradOut, input, pre)
	case core.LayerMultiHeadAttention:
		ml, ok := cell.Op.(*mha.Layer)
		if !ok {
			return nil, nil, fmt.Errorf("mha op %T", cell.Op)
		}
		return mha.Backward(ml, gradOut, input, pre)
	case core.LayerSwiGLU:
		sl, ok := cell.Op.(*swiglu.Layer)
		if !ok {
			return nil, nil, fmt.Errorf("swiglu op %T", cell.Op)
		}
		return swiglu.Backward(sl, gradOut, input, pre)
	case core.LayerRMSNorm:
		rl, ok := cell.Op.(*rmsnorm.Layer)
		if !ok {
			return nil, nil, fmt.Errorf("rmsnorm op %T", cell.Op)
		}
		return rmsnorm.Backward(rl, gradOut, input, pre)
	case core.LayerLayerNorm:
		ll, ok := cell.Op.(*layernorm.Layer)
		if !ok {
			return nil, nil, fmt.Errorf("layernorm op %T", cell.Op)
		}
		return layernorm.Backward(ll, gradOut, input, pre)
	case core.LayerCNN1:
		cl, ok := cell.Op.(*cnn1.Layer)
		if !ok {
			return nil, nil, fmt.Errorf("cnn1 op %T", cell.Op)
		}
		return cnn1.Backward(cl, gradOut, input, pre)
	case core.LayerCNN2:
		cl, ok := cell.Op.(*cnn2.Layer)
		if !ok {
			return nil, nil, fmt.Errorf("cnn2 op %T", cell.Op)
		}
		return cnn2.Backward(cl, gradOut, input, pre)
	case core.LayerCNN3:
		cl, ok := cell.Op.(*cnn3.Layer)
		if !ok {
			return nil, nil, fmt.Errorf("cnn3 op %T", cell.Op)
		}
		return cnn3.Backward(cl, gradOut, input, pre)
	case core.LayerRNN:
		rl, ok := cell.Op.(*rnn.Layer)
		if !ok {
			return nil, nil, fmt.Errorf("rnn op %T", cell.Op)
		}
		return rnn.Backward(rl, gradOut, input, pre)
	case core.LayerLSTM:
		ll, ok := cell.Op.(*lstm.Layer)
		if !ok {
			return nil, nil, fmt.Errorf("lstm op %T", cell.Op)
		}
		return lstm.Backward(ll, gradOut, input, pre)
	case core.LayerEmbedding:
		el, ok := cell.Op.(*embedding.Layer)
		if !ok {
			return nil, nil, fmt.Errorf("embedding op %T", cell.Op)
		}
		return embedding.Backward(el, gradOut, input, pre)
	case core.LayerSequential:
		sl, ok := cell.Op.(*sequential.Layer)
		if !ok {
			return nil, nil, fmt.Errorf("sequential op %T", cell.Op)
		}
		return sequential.Backward(sl, gradOut, input, pre)
	case core.LayerResidual:
		rl, ok := cell.Op.(*residual.Layer)
		if !ok {
			return nil, nil, fmt.Errorf("residual op %T", cell.Op)
		}
		return residual.Backward(rl, gradOut, input, pre)
	case core.LayerSoftmax:
		sl, ok := cell.Op.(*softmax.Layer)
		if !ok {
			return nil, nil, fmt.Errorf("softmax op %T", cell.Op)
		}
		return softmax.Backward(sl, gradOut, input, pre)
	default:
		return nil, nil, fmt.Errorf("tween: backward unsupported type %s", cell.Layer.Type)
	}
}
