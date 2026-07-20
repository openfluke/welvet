package evolution

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/cnn2"
	"github.com/openfluke/welvet/layers/cnn3"
	"github.com/openfluke/welvet/layers/convt1"
	"github.com/openfluke/welvet/layers/convt2"
	"github.com/openfluke/welvet/layers/convt3"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/embedding"
	"github.com/openfluke/welvet/layers/kmeans"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/lstm"
	"github.com/openfluke/welvet/layers/mamba"
	"github.com/openfluke/welvet/layers/metacognition"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/rnn"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/layers/softmax"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/weights"
)

func cloneOp(op any) (any, error) {
	if op == nil {
		return nil, nil
	}
	switch v := op.(type) {
	case *dense.Layer:
		return cloneDense(v)
	case *softmax.Layer:
		cp := *v
		return &cp, nil
	case *mha.Layer:
		return cloneMHA(v)
	case *swiglu.Layer:
		return cloneSwiGLU(v)
	case *rmsnorm.Layer:
		return cloneRMSNorm(v)
	case *layernorm.Layer:
		return cloneLayerNorm(v)
	case *cnn1.Layer:
		return cloneCNN1(v)
	case *cnn2.Layer:
		return cloneCNN2(v)
	case *cnn3.Layer:
		return cloneCNN3(v)
	case *convt1.Layer:
		return cloneConvT1(v)
	case *convt2.Layer:
		return cloneConvT2(v)
	case *convt3.Layer:
		return cloneConvT3(v)
	case *rnn.Layer:
		return cloneRNN(v)
	case *lstm.Layer:
		return cloneLSTM(v)
	case *embedding.Layer:
		return cloneEmbedding(v)
	case *sequential.Layer:
		return cloneSequential(v)
	case *residual.Layer:
		return cloneResidual(v)
	case *parallel.Layer:
		return cloneParallel(v)
	case *kmeans.Layer:
		return cloneKMeans(v)
	case *mamba.Layer:
		return cloneMamba(v)
	case *metacognition.Layer:
		return cloneMetacognition(v)
	default:
		return nil, fmt.Errorf("evolution: clone unsupported Op %T", op)
	}
}

func storeMeta(s *weights.Store, fallbackDT core.DType) (core.DType, quant.Format) {
	if s == nil {
		return fallbackDT, quant.FormatNone
	}
	return s.DType, s.Format
}

func flattenOrNil(s *weights.Store) ([]float32, error) {
	if s == nil {
		return nil, nil
	}
	return s.FlattenF32()
}

func cloneDense(src *dense.Layer) (*dense.Layer, error) {
	if src == nil {
		return nil, nil
	}
	init, err := flattenOrNil(src.Weights)
	if err != nil {
		return nil, err
	}
	dt, format := storeMeta(src.Weights, src.Core.DType)
	dst, err := dense.NewConfigured(
		src.Core.InputHeight, src.Core.OutputHeight,
		src.Core.Activation, dt, format, init,
	)
	if err != nil {
		return nil, err
	}
	dst.Exec = src.Exec
	dst.Core = src.Core
	if src.Weights != nil && len(src.Weights.Bias) > 0 {
		dst.Weights.Bias = append([]float64(nil), src.Weights.Bias...)
	}
	return dst, nil
}

func cloneMHA(src *mha.Layer) (*mha.Layer, error) {
	if src == nil {
		return nil, nil
	}
	q, err := flattenOrNil(src.Q.Weights)
	if err != nil {
		return nil, err
	}
	k, err := flattenOrNil(src.K.Weights)
	if err != nil {
		return nil, err
	}
	v, err := flattenOrNil(src.V.Weights)
	if err != nil {
		return nil, err
	}
	o, err := flattenOrNil(src.O.Weights)
	if err != nil {
		return nil, err
	}
	dt, format := storeMeta(src.Q.Weights, src.Core.DType)
	dst, err := mha.NewConfigured(src.Cfg, dt, format, q, k, v, o)
	if err != nil {
		return nil, err
	}
	dst.Exec = src.Exec
	dst.Core = src.Core
	dst.QNormWeight = append([]float64(nil), src.QNormWeight...)
	dst.KNormWeight = append([]float64(nil), src.KNormWeight...)
	copyDenseBias(src.Q, dst.Q)
	copyDenseBias(src.K, dst.K)
	copyDenseBias(src.V, dst.V)
	copyDenseBias(src.O, dst.O)
	return dst, nil
}

func cloneSwiGLU(src *swiglu.Layer) (*swiglu.Layer, error) {
	if src == nil {
		return nil, nil
	}
	g, err := flattenOrNil(src.Gate.Weights)
	if err != nil {
		return nil, err
	}
	u, err := flattenOrNil(src.Up.Weights)
	if err != nil {
		return nil, err
	}
	d, err := flattenOrNil(src.Down.Weights)
	if err != nil {
		return nil, err
	}
	dt, format := storeMeta(src.Gate.Weights, src.Core.DType)
	dst, err := swiglu.NewConfigured(src.Cfg, dt, format, g, u, d)
	if err != nil {
		return nil, err
	}
	dst.Exec = src.Exec
	dst.Core = src.Core
	copyDenseBias(src.Gate, dst.Gate)
	copyDenseBias(src.Up, dst.Up)
	copyDenseBias(src.Down, dst.Down)
	return dst, nil
}

func cloneRMSNorm(src *rmsnorm.Layer) (*rmsnorm.Layer, error) {
	if src == nil {
		return nil, nil
	}
	g, err := flattenOrNil(src.Gamma)
	if err != nil {
		return nil, err
	}
	dt, format := storeMeta(src.Gamma, src.Core.DType)
	dst, err := rmsnorm.NewConfigured(src.Cfg, dt, format, g)
	if err != nil {
		return nil, err
	}
	dst.Exec = src.Exec
	dst.Core = src.Core
	return dst, nil
}

func cloneLayerNorm(src *layernorm.Layer) (*layernorm.Layer, error) {
	if src == nil {
		return nil, nil
	}
	g, err := flattenOrNil(src.Gamma)
	if err != nil {
		return nil, err
	}
	b, err := flattenOrNil(src.Beta)
	if err != nil {
		return nil, err
	}
	dt, format := storeMeta(src.Gamma, src.Core.DType)
	dst, err := layernorm.NewConfigured(src.Cfg, dt, format, g, b)
	if err != nil {
		return nil, err
	}
	dst.Exec = src.Exec
	dst.Core = src.Core
	return dst, nil
}

func cloneCNN1(src *cnn1.Layer) (*cnn1.Layer, error) {
	if src == nil {
		return nil, nil
	}
	w, err := flattenOrNil(src.Proj.Weights)
	if err != nil {
		return nil, err
	}
	dt, format := storeMeta(src.Proj.Weights, src.Core.DType)
	dst, err := cnn1.NewConfigured(src.Cfg, dt, format, w)
	if err != nil {
		return nil, err
	}
	dst.Exec = src.Exec
	dst.Core = src.Core
	copyDenseBias(src.Proj, dst.Proj)
	return dst, nil
}

func cloneCNN2(src *cnn2.Layer) (*cnn2.Layer, error) {
	if src == nil {
		return nil, nil
	}
	w, err := flattenOrNil(src.Proj.Weights)
	if err != nil {
		return nil, err
	}
	dt, format := storeMeta(src.Proj.Weights, src.Core.DType)
	dst, err := cnn2.NewConfigured(src.Cfg, dt, format, w)
	if err != nil {
		return nil, err
	}
	dst.Exec = src.Exec
	dst.Core = src.Core
	copyDenseBias(src.Proj, dst.Proj)
	return dst, nil
}

func cloneCNN3(src *cnn3.Layer) (*cnn3.Layer, error) {
	if src == nil {
		return nil, nil
	}
	w, err := flattenOrNil(src.Proj.Weights)
	if err != nil {
		return nil, err
	}
	dt, format := storeMeta(src.Proj.Weights, src.Core.DType)
	dst, err := cnn3.NewConfigured(src.Cfg, dt, format, w)
	if err != nil {
		return nil, err
	}
	dst.Exec = src.Exec
	dst.Core = src.Core
	copyDenseBias(src.Proj, dst.Proj)
	return dst, nil
}

func cloneRNN(src *rnn.Layer) (*rnn.Layer, error) {
	if src == nil {
		return nil, nil
	}
	ih, err := flattenOrNil(src.IH.Weights)
	if err != nil {
		return nil, err
	}
	hh, err := flattenOrNil(src.HH.Weights)
	if err != nil {
		return nil, err
	}
	h := src.Cfg.HiddenSize
	packed := make([]float32, 0, len(ih)+len(hh)+h)
	packed = append(packed, ih...)
	packed = append(packed, hh...)
	if src.IH.Weights != nil && len(src.IH.Weights.Bias) > 0 {
		for _, b := range src.IH.Weights.Bias {
			packed = append(packed, float32(b))
		}
	} else {
		packed = append(packed, make([]float32, h)...)
	}
	dt, format := storeMeta(src.IH.Weights, src.Core.DType)
	dst, err := rnn.NewConfigured(src.Cfg, dt, format, packed)
	if err != nil {
		return nil, err
	}
	dst.Exec = src.Exec
	dst.Core = src.Core
	return dst, nil
}

func cloneLSTM(src *lstm.Layer) (*lstm.Layer, error) {
	if src == nil {
		return nil, nil
	}
	gates := []*lstm.Gate{src.I, src.F, src.G, src.O}
	var packed []float32
	for _, g := range gates {
		if g == nil {
			continue
		}
		ih, err := flattenOrNil(g.IH.Weights)
		if err != nil {
			return nil, err
		}
		hh, err := flattenOrNil(g.HH.Weights)
		if err != nil {
			return nil, err
		}
		packed = append(packed, ih...)
		packed = append(packed, hh...)
		h := src.Cfg.HiddenSize
		if g.IH.Weights != nil && len(g.IH.Weights.Bias) >= h {
			for i := 0; i < h; i++ {
				packed = append(packed, float32(g.IH.Weights.Bias[i]))
			}
		} else {
			packed = append(packed, make([]float32, h)...)
		}
	}
	dt, format := storeMeta(src.I.IH.Weights, src.Core.DType)
	dst, err := lstm.NewConfigured(src.Cfg, dt, format, packed)
	if err != nil {
		return nil, err
	}
	dst.Exec = src.Exec
	dst.Core = src.Core
	return dst, nil
}

func cloneEmbedding(src *embedding.Layer) (*embedding.Layer, error) {
	if src == nil {
		return nil, nil
	}
	w, err := flattenOrNil(src.Weights)
	if err != nil {
		return nil, err
	}
	dt, format := storeMeta(src.Weights, src.Core.DType)
	dst, err := embedding.NewConfigured(src.Cfg, dt, format, w)
	if err != nil {
		return nil, err
	}
	dst.Exec = src.Exec
	dst.Core = src.Core
	return dst, nil
}

func cloneSequential(src *sequential.Layer) (*sequential.Layer, error) {
	if src == nil {
		return nil, nil
	}
	dim := src.Cfg.Dim
	childN := dim * dim
	packed := make([]float32, 0, len(src.Children)*childN)
	for _, ch := range src.Children {
		w, err := flattenOrNil(ch.Weights)
		if err != nil {
			return nil, err
		}
		if len(w) < childN {
			tmp := make([]float32, childN)
			copy(tmp, w)
			w = tmp
		}
		packed = append(packed, w[:childN]...)
	}
	dt, format := src.Core.DType, quant.FormatNone
	if len(src.Children) > 0 && src.Children[0].Weights != nil {
		dt, format = storeMeta(src.Children[0].Weights, src.Core.DType)
	}
	dst, err := sequential.NewConfigured(src.Cfg, dt, format, packed)
	if err != nil {
		return nil, err
	}
	dst.Exec = src.Exec
	dst.Core = src.Core
	return dst, nil
}

func cloneResidual(src *residual.Layer) (*residual.Layer, error) {
	if src == nil {
		return nil, nil
	}
	dim := src.Cfg.Dim
	childN := dim * dim
	packed := make([]float32, 0, len(src.Children)*childN)
	for _, ch := range src.Children {
		w, err := flattenOrNil(ch.Weights)
		if err != nil {
			return nil, err
		}
		if len(w) < childN {
			tmp := make([]float32, childN)
			copy(tmp, w)
			w = tmp
		}
		packed = append(packed, w[:childN]...)
	}
	dt, format := src.Core.DType, quant.FormatNone
	if len(src.Children) > 0 && src.Children[0].Weights != nil {
		dt, format = storeMeta(src.Children[0].Weights, src.Core.DType)
	}
	dst, err := residual.NewConfigured(src.Cfg, dt, format, packed)
	if err != nil {
		return nil, err
	}
	dst.Exec = src.Exec
	dst.Core = src.Core
	return dst, nil
}

func cloneConvT1(src *convt1.Layer) (*convt1.Layer, error) {
	if src == nil {
		return nil, nil
	}
	w, err := flattenOrNil(src.Proj.Weights)
	if err != nil {
		return nil, err
	}
	dt, format := storeMeta(src.Proj.Weights, src.Core.DType)
	dst, err := convt1.NewConfigured(src.Cfg, dt, format, w)
	if err != nil {
		return nil, err
	}
	dst.Exec = src.Exec
	dst.Core = src.Core
	copyDenseBias(src.Proj, dst.Proj)
	return dst, nil
}

func cloneConvT2(src *convt2.Layer) (*convt2.Layer, error) {
	if src == nil {
		return nil, nil
	}
	w, err := flattenOrNil(src.Proj.Weights)
	if err != nil {
		return nil, err
	}
	dt, format := storeMeta(src.Proj.Weights, src.Core.DType)
	dst, err := convt2.NewConfigured(src.Cfg, dt, format, w)
	if err != nil {
		return nil, err
	}
	dst.Exec = src.Exec
	dst.Core = src.Core
	copyDenseBias(src.Proj, dst.Proj)
	return dst, nil
}

func cloneConvT3(src *convt3.Layer) (*convt3.Layer, error) {
	if src == nil {
		return nil, nil
	}
	w, err := flattenOrNil(src.Proj.Weights)
	if err != nil {
		return nil, err
	}
	dt, format := storeMeta(src.Proj.Weights, src.Core.DType)
	dst, err := convt3.NewConfigured(src.Cfg, dt, format, w)
	if err != nil {
		return nil, err
	}
	dst.Exec = src.Exec
	dst.Core = src.Core
	copyDenseBias(src.Proj, dst.Proj)
	return dst, nil
}

func cloneKMeans(src *kmeans.Layer) (*kmeans.Layer, error) {
	if src == nil {
		return nil, nil
	}
	w, err := flattenOrNil(src.Centers.Weights)
	if err != nil {
		return nil, err
	}
	dt, format := storeMeta(src.Centers.Weights, src.Core.DType)
	dst, err := kmeans.NewConfigured(src.Cfg, dt, format, w)
	if err != nil {
		return nil, err
	}
	dst.Exec = src.Exec
	dst.Core = src.Core
	copyDenseBias(src.Centers, dst.Centers)
	return dst, nil
}

func cloneMetacognition(src *metacognition.Layer) (*metacognition.Layer, error) {
	if src == nil {
		return nil, nil
	}
	w, err := flattenOrNil(src.Observed.Weights)
	if err != nil {
		return nil, err
	}
	dt, format := storeMeta(src.Observed.Weights, src.Core.DType)
	dst, err := metacognition.NewConfigured(src.Cfg, dt, format, w)
	if err != nil {
		return nil, err
	}
	dst.Exec = src.Exec
	dst.Core = src.Core
	dst.Rules = append([]metacognition.Rule(nil), src.Rules...)
	copyDenseBias(src.Observed, dst.Observed)
	return dst, nil
}

func cloneParallel(src *parallel.Layer) (*parallel.Layer, error) {
	if src == nil {
		return nil, nil
	}
	childN := src.Cfg.OutFeat * src.Cfg.Dim
	packed := make([]float32, 0, len(src.Branches)*childN)
	for _, ch := range src.Branches {
		w, err := flattenOrNil(ch.Weights)
		if err != nil {
			return nil, err
		}
		if len(w) < childN {
			tmp := make([]float32, childN)
			copy(tmp, w)
			w = tmp
		}
		packed = append(packed, w[:childN]...)
	}
	var gateInit []float32
	if src.Gate != nil {
		g, err := flattenOrNil(src.Gate.Weights)
		if err != nil {
			return nil, err
		}
		gateInit = g
	}
	dt, format := src.Core.DType, quant.FormatNone
	if len(src.Branches) > 0 && src.Branches[0].Weights != nil {
		dt, format = storeMeta(src.Branches[0].Weights, src.Core.DType)
	}
	dst, err := parallel.NewConfigured(src.Cfg, dt, format, packed, gateInit)
	if err != nil {
		return nil, err
	}
	dst.Exec = src.Exec
	dst.Core = src.Core
	for i := range src.Branches {
		copyDenseBias(src.Branches[i], dst.Branches[i])
	}
	copyDenseBias(src.Gate, dst.Gate)
	return dst, nil
}

func cloneMamba(src *mamba.Layer) (*mamba.Layer, error) {
	if src == nil {
		return nil, nil
	}
	inW, err := flattenOrNil(src.InProj.Weights)
	if err != nil {
		return nil, err
	}
	outW, err := flattenOrNil(src.OutProj.Weights)
	if err != nil {
		return nil, err
	}
	dt, format := storeMeta(src.InProj.Weights, src.Core.DType)
	dst, err := mamba.NewConfigured(src.Cfg, dt, format, inW, outW,
		append([]float32(nil), src.ALog...), append([]float32(nil), src.DSkip...))
	if err != nil {
		return nil, err
	}
	dst.Exec = src.Exec
	dst.Core = src.Core
	copyDenseBias(src.InProj, dst.InProj)
	copyDenseBias(src.OutProj, dst.OutProj)
	return dst, nil
}

func copyDenseBias(src, dst *dense.Layer) {
	if src == nil || dst == nil || src.Weights == nil || dst.Weights == nil {
		return
	}
	if len(src.Weights.Bias) > 0 {
		dst.Weights.Bias = append([]float64(nil), src.Weights.Bias...)
	}
}
