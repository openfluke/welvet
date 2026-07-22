package serialization

import (
	"fmt"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/cnn2"
	"github.com/openfluke/welvet/layers/cnn3"
	"github.com/openfluke/welvet/layers/convt1"
	"github.com/openfluke/welvet/layers/convt2"
	"github.com/openfluke/welvet/layers/convt3"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/embedding"
	"github.com/openfluke/welvet/layers/gdn"
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
	"github.com/openfluke/welvet/weights"
)

func encodeCell(c architecture.Coord, cell *architecture.Cell) (LayerSpec, error) {
	ls := LayerSpec{
		Z: c.Z, Y: c.Y, X: c.X, L: c.L,
		Type:         cell.Layer.Type.String(),
		Activation:   cell.Layer.Activation.String(),
		DType:        cell.Layer.DType.String(),
		IsDisabled:   cell.Layer.IsDisabled,
		IsRemoteLink: cell.IsRemoteLink,
		TargetZ:      cell.TargetZ,
		TargetY:      cell.TargetY,
		TargetX:      cell.TargetX,
		TargetL:      cell.TargetL,
		InputHeight:  cell.Layer.InputHeight,
		OutputHeight: cell.Layer.OutputHeight,
	}
	if cell.Op == nil {
		return ls, nil
	}
	switch op := cell.Op.(type) {
	case *dense.Layer:
		return encodeDense(ls, op)
	case *mha.Layer:
		return encodeMHA(ls, op)
	case *swiglu.Layer:
		return encodeSwiGLU(ls, op)
	case *rmsnorm.Layer:
		return encodeRMSNorm(ls, op)
	case *layernorm.Layer:
		return encodeLayerNorm(ls, op)
	case *cnn1.Layer:
		return encodeCNN1(ls, op)
	case *cnn2.Layer:
		return encodeCNN2(ls, op)
	case *cnn3.Layer:
		return encodeCNN3(ls, op)
	case *convt1.Layer:
		return encodeConvT1(ls, op)
	case *convt2.Layer:
		return encodeConvT2(ls, op)
	case *convt3.Layer:
		return encodeConvT3(ls, op)
	case *rnn.Layer:
		return encodeRNN(ls, op)
	case *lstm.Layer:
		return encodeLSTM(ls, op)
	case *embedding.Layer:
		return encodeEmbedding(ls, op)
	case *softmax.Layer:
		return encodeSoftmax(ls, op)
	case *kmeans.Layer:
		return encodeKMeans(ls, op)
	case *parallel.Layer:
		return encodeParallel(ls, op)
	case *sequential.Layer:
		return encodeSequential(ls, op)
	case *residual.Layer:
		return encodeResidual(ls, op)
	case *metacognition.Layer:
		return encodeMeta(ls, op)
	case *mamba.Layer:
		return encodeMamba(ls, op)
	case *gdn.Layer:
		return encodeGDN(ls, op)
	default:
		return ls, fmt.Errorf("serialization: unsupported Op %T", cell.Op)
	}
}

func addStore(ls *LayerSpec, name string, s *weights.Store) error {
	if s == nil {
		return nil
	}
	b, err := encodeStore(name, s)
	if err != nil {
		return err
	}
	ls.Stores = append(ls.Stores, b)
	if ls.Format == "" {
		ls.Format = b.Format
	}
	return nil
}

func addDenseStore(ls *LayerSpec, name string, d *dense.Layer) error {
	if d == nil {
		return nil
	}
	return addStore(ls, name, d.Weights)
}

func putExtra(ls *LayerSpec, name string, v []float32) error {
	if len(v) == 0 {
		return nil
	}
	enc, err := encodeF32Extra(name, v)
	if err != nil {
		return err
	}
	if ls.Extras == nil {
		ls.Extras = map[string]string{}
	}
	ls.Extras[name] = enc
	return nil
}

func encodeDense(ls LayerSpec, op *dense.Layer) (LayerSpec, error) {
	if op.Weights != nil {
		ls.InputHeight = op.Weights.Cols
		ls.OutputHeight = op.Weights.Rows
		if err := addStore(&ls, "w", op.Weights); err != nil {
			return ls, err
		}
	} else {
		ls.InputHeight = op.Core.InputHeight
		ls.OutputHeight = op.Core.OutputHeight
	}
	return ls, nil
}

func encodeMHA(ls LayerSpec, op *mha.Layer) (LayerSpec, error) {
	ls.DModel = op.Cfg.DModel
	ls.NumHeads = op.Cfg.NumHeads
	ls.NumKVHeads = op.Cfg.NumKVHeads
	ls.HeadDim = op.Cfg.HeadDim
	ls.SeqLength = op.Cfg.MaxSeqLen
	ls.Mask = op.Cfg.Mask.String()
	ls.Causal = op.Cfg.Causal
	ls.InputHeight = op.Cfg.DModel
	ls.OutputHeight = op.Cfg.DModel
	for _, pair := range []struct {
		n string
		d *dense.Layer
	}{
		{"q", op.Q}, {"k", op.K}, {"v", op.V}, {"o", op.O},
	} {
		if err := addDenseStore(&ls, pair.n, pair.d); err != nil {
			return ls, err
		}
	}
	qn := make([]float32, len(op.QNormWeight))
	for i, v := range op.QNormWeight {
		qn[i] = float32(v)
	}
	kn := make([]float32, len(op.KNormWeight))
	for i, v := range op.KNormWeight {
		kn[i] = float32(v)
	}
	if err := putExtra(&ls, "q_norm", qn); err != nil {
		return ls, err
	}
	if err := putExtra(&ls, "k_norm", kn); err != nil {
		return ls, err
	}
	return ls, nil
}

func encodeSwiGLU(ls LayerSpec, op *swiglu.Layer) (LayerSpec, error) {
	ls.InputHeight = op.Cfg.InputDim
	ls.OutputHeight = op.Cfg.InputDim
	ls.IntermediateDim = op.Cfg.IntermediateDim
	for _, pair := range []struct {
		n string
		d *dense.Layer
	}{
		{"gate", op.Gate}, {"up", op.Up}, {"down", op.Down},
	} {
		if err := addDenseStore(&ls, pair.n, pair.d); err != nil {
			return ls, err
		}
	}
	return ls, nil
}

func encodeRMSNorm(ls LayerSpec, op *rmsnorm.Layer) (LayerSpec, error) {
	ls.InputHeight = op.Cfg.Dim
	ls.OutputHeight = op.Cfg.Dim
	ls.Eps = op.Cfg.Eps
	return ls, addStore(&ls, "gamma", op.Gamma)
}

func encodeLayerNorm(ls LayerSpec, op *layernorm.Layer) (LayerSpec, error) {
	ls.InputHeight = op.Cfg.Dim
	ls.OutputHeight = op.Cfg.Dim
	ls.Eps = op.Cfg.Eps
	if err := addStore(&ls, "gamma", op.Gamma); err != nil {
		return ls, err
	}
	return ls, addStore(&ls, "beta", op.Beta)
}

func encodeCNN1(ls LayerSpec, op *cnn1.Layer) (LayerSpec, error) {
	ls.InputChannels = op.Cfg.InChannels
	ls.Filters = op.Cfg.Filters
	ls.SeqLength = op.Cfg.SeqLen
	ls.KernelSize = op.Cfg.Kernel
	ls.Stride = op.Cfg.Stride
	ls.Padding = op.Cfg.Padding
	ls.InputHeight = op.Cfg.SeqLen
	ls.OutputHeight = op.Cfg.OutLen()
	return ls, addDenseStore(&ls, "proj", op.Proj)
}

func encodeCNN2(ls LayerSpec, op *cnn2.Layer) (LayerSpec, error) {
	ls.InputChannels = op.Cfg.InChannels
	ls.Filters = op.Cfg.Filters
	ls.InputHeight = op.Cfg.Height
	ls.InputWidth = op.Cfg.Width
	ls.KernelSize = op.Cfg.Kernel
	ls.Stride = op.Cfg.Stride
	ls.Padding = op.Cfg.Padding
	ls.OutputHeight = op.Cfg.OutH()
	ls.OutputWidth = op.Cfg.OutW()
	return ls, addDenseStore(&ls, "proj", op.Proj)
}

func encodeCNN3(ls LayerSpec, op *cnn3.Layer) (LayerSpec, error) {
	ls.InputChannels = op.Cfg.InChannels
	ls.Filters = op.Cfg.Filters
	ls.InputDepth = op.Cfg.Depth
	ls.InputHeight = op.Cfg.Height
	ls.InputWidth = op.Cfg.Width
	ls.KernelSize = op.Cfg.Kernel
	ls.Stride = op.Cfg.Stride
	ls.Padding = op.Cfg.Padding
	ls.OutputDepth = op.Cfg.OutD()
	ls.OutputHeight = op.Cfg.OutH()
	ls.OutputWidth = op.Cfg.OutW()
	return ls, addDenseStore(&ls, "proj", op.Proj)
}

func encodeConvT1(ls LayerSpec, op *convt1.Layer) (LayerSpec, error) {
	ls.InputChannels = op.Cfg.InChannels
	ls.Filters = op.Cfg.Filters
	ls.SeqLength = op.Cfg.SeqLen
	ls.KernelSize = op.Cfg.Kernel
	ls.Stride = op.Cfg.Stride
	ls.Padding = op.Cfg.Padding
	ls.OutputPadding = op.Cfg.OutputPadding
	ls.InputHeight = op.Cfg.SeqLen
	ls.OutputHeight = op.Cfg.OutLen()
	return ls, addDenseStore(&ls, "proj", op.Proj)
}

func encodeConvT2(ls LayerSpec, op *convt2.Layer) (LayerSpec, error) {
	ls.InputChannels = op.Cfg.InChannels
	ls.Filters = op.Cfg.Filters
	ls.InputHeight = op.Cfg.Height
	ls.InputWidth = op.Cfg.Width
	ls.KernelSize = op.Cfg.Kernel
	ls.Stride = op.Cfg.Stride
	ls.Padding = op.Cfg.Padding
	ls.OutputPadding = op.Cfg.OutputPadding
	ls.OutputHeight = op.Cfg.OutH()
	ls.OutputWidth = op.Cfg.OutW()
	return ls, addDenseStore(&ls, "proj", op.Proj)
}

func encodeConvT3(ls LayerSpec, op *convt3.Layer) (LayerSpec, error) {
	ls.InputChannels = op.Cfg.InChannels
	ls.Filters = op.Cfg.Filters
	ls.InputDepth = op.Cfg.Depth
	ls.InputHeight = op.Cfg.Height
	ls.InputWidth = op.Cfg.Width
	ls.KernelSize = op.Cfg.Kernel
	ls.Stride = op.Cfg.Stride
	ls.Padding = op.Cfg.Padding
	ls.OutputPadding = op.Cfg.OutputPadding
	ls.OutputDepth = op.Cfg.OutD()
	ls.OutputHeight = op.Cfg.OutH()
	ls.OutputWidth = op.Cfg.OutW()
	return ls, addDenseStore(&ls, "proj", op.Proj)
}

func encodeRNN(ls LayerSpec, op *rnn.Layer) (LayerSpec, error) {
	ls.InputHeight = op.Cfg.InputSize
	ls.OutputHeight = op.Cfg.HiddenSize
	ls.SeqLength = op.Cfg.SeqLen
	if err := addDenseStore(&ls, "ih", op.IH); err != nil {
		return ls, err
	}
	return ls, addDenseStore(&ls, "hh", op.HH)
}

func encodeLSTM(ls LayerSpec, op *lstm.Layer) (LayerSpec, error) {
	ls.InputHeight = op.Cfg.InputSize
	ls.OutputHeight = op.Cfg.HiddenSize
	ls.SeqLength = op.Cfg.SeqLen
	gates := []struct {
		p string
		g *lstm.Gate
	}{
		{"i", op.I}, {"f", op.F}, {"g", op.G}, {"o", op.O},
	}
	for _, g := range gates {
		if g.g == nil {
			continue
		}
		if err := addDenseStore(&ls, g.p+"_ih", g.g.IH); err != nil {
			return ls, err
		}
		if err := addDenseStore(&ls, g.p+"_hh", g.g.HH); err != nil {
			return ls, err
		}
	}
	return ls, nil
}

func encodeEmbedding(ls LayerSpec, op *embedding.Layer) (LayerSpec, error) {
	ls.VocabSize = op.Cfg.VocabSize
	ls.EmbeddingDim = op.Cfg.EmbeddingDim
	ls.SeqLength = op.Cfg.SeqLen
	ls.InputHeight = op.Cfg.VocabSize
	ls.OutputHeight = op.Cfg.EmbeddingDim
	return ls, addStore(&ls, "w", op.Weights)
}

func encodeSoftmax(ls LayerSpec, op *softmax.Layer) (LayerSpec, error) {
	ls.InputHeight = op.Cfg.Dim
	ls.OutputHeight = op.Cfg.Dim
	ls.SeqLength = op.Cfg.SeqLen
	ls.Temperature = op.Cfg.Temperature
	ls.SoftmaxKind = op.Cfg.Kind.String()
	ls.SoftmaxRows = op.Cfg.Rows
	ls.SoftmaxCols = op.Cfg.Cols
	ls.SoftmaxMask = append([]bool(nil), op.Cfg.Mask...)
	ls.EntmaxAlpha = op.Cfg.EntmaxAlpha
	ls.HierarchyLevels = append([]int(nil), op.Cfg.HierarchyLevels...)
	return ls, nil
}

func encodeKMeans(ls LayerSpec, op *kmeans.Layer) (LayerSpec, error) {
	ls.NumClusters = op.Cfg.NumClusters
	ls.InputHeight = op.Cfg.FeatureDim
	ls.OutputHeight = op.Cfg.OutDim()
	ls.Temperature = op.Cfg.Temperature
	ls.OutputMode = string(op.Cfg.OutputMode)
	return ls, addDenseStore(&ls, "centers", op.Centers)
}

func encodeParallel(ls LayerSpec, op *parallel.Layer) (LayerSpec, error) {
	ls.InputHeight = op.Cfg.Dim
	out := op.Cfg.OutDim()
	if out == 0 {
		out = op.Core.OutputHeight
	}
	ls.OutputHeight = out
	ls.Branches = op.Cfg.Branches
	ls.OutFeat = op.Cfg.OutFeat
	ls.CombineMode = string(op.Cfg.Combine)
	ls.SeqLength = op.Cfg.SeqLen
	allDense := true
	for _, b := range op.Branches {
		if _, ok := b.(*dense.Layer); !ok {
			allDense = false
			break
		}
	}
	if allDense {
		for i, b := range op.Branches {
			if err := addDenseStore(&ls, fmt.Sprintf("branch.%d", i), b.(*dense.Layer)); err != nil {
				return ls, err
			}
		}
		return ls, addDenseStore(&ls, "gate", op.Gate)
	}
	ls.BranchOps = make([]LayerSpec, len(op.Branches))
	for i, b := range op.Branches {
		bls, err := encodeBranchOp(b)
		if err != nil {
			return ls, fmt.Errorf("parallel branch %d: %w", i, err)
		}
		ls.BranchOps[i] = bls
	}
	return ls, addDenseStore(&ls, "gate", op.Gate)
}

func encodeBranchOp(op any) (LayerSpec, error) {
	if op == nil {
		return LayerSpec{}, fmt.Errorf("serialization: nil branch Op")
	}
	ls := LayerSpec{}
	switch v := op.(type) {
	case *dense.Layer:
		ls.Type = "Dense"
		ls.Activation = v.Core.Activation.String()
		ls.DType = v.Core.DType.String()
		return encodeDense(ls, v)
	case *mha.Layer:
		ls.Type = "MultiHeadAttention"
		ls.DType = v.Core.DType.String()
		return encodeMHA(ls, v)
	case *swiglu.Layer:
		ls.Type = "SwiGLU"
		ls.DType = v.Core.DType.String()
		return encodeSwiGLU(ls, v)
	case *rmsnorm.Layer:
		ls.Type = "RMSNorm"
		ls.DType = v.Core.DType.String()
		return encodeRMSNorm(ls, v)
	case *layernorm.Layer:
		ls.Type = "LayerNorm"
		ls.DType = v.Core.DType.String()
		return encodeLayerNorm(ls, v)
	case *softmax.Layer:
		ls.Type = "Softmax"
		return encodeSoftmax(ls, v)
	case *cnn1.Layer:
		ls.Type = "CNN1"
		ls.Activation = v.Core.Activation.String()
		return encodeCNN1(ls, v)
	case *cnn2.Layer:
		ls.Type = "CNN2"
		ls.Activation = v.Core.Activation.String()
		return encodeCNN2(ls, v)
	case *cnn3.Layer:
		ls.Type = "CNN3"
		ls.Activation = v.Core.Activation.String()
		return encodeCNN3(ls, v)
	case *convt1.Layer:
		ls.Type = "ConvTransposed1D"
		ls.Activation = v.Core.Activation.String()
		return encodeConvT1(ls, v)
	case *convt2.Layer:
		ls.Type = "ConvTransposed2D"
		ls.Activation = v.Core.Activation.String()
		return encodeConvT2(ls, v)
	case *convt3.Layer:
		ls.Type = "ConvTransposed3D"
		ls.Activation = v.Core.Activation.String()
		return encodeConvT3(ls, v)
	case *rnn.Layer:
		ls.Type = "RNN"
		return encodeRNN(ls, v)
	case *lstm.Layer:
		ls.Type = "LSTM"
		return encodeLSTM(ls, v)
	case *embedding.Layer:
		ls.Type = "Embedding"
		return encodeEmbedding(ls, v)
	case *kmeans.Layer:
		ls.Type = "KMeans"
		ls.Activation = v.Core.Activation.String()
		return encodeKMeans(ls, v)
	case *parallel.Layer:
		ls.Type = "Parallel"
		ls.DType = v.Core.DType.String()
		return encodeParallel(ls, v)
	case *sequential.Layer:
		ls.Type = "Sequential"
		ls.DType = v.Core.DType.String()
		return encodeSequential(ls, v)
	case *residual.Layer:
		ls.Type = "Residual"
		ls.DType = v.Core.DType.String()
		return encodeResidual(ls, v)
	case *metacognition.Layer:
		ls.Type = "Metacognition"
		return encodeMeta(ls, v)
	case *mamba.Layer:
		ls.Type = "Mamba"
		return encodeMamba(ls, v)
	case *gdn.Layer:
		ls.Type = "GDN"
		return encodeGDN(ls, v)
	default:
		return ls, fmt.Errorf("serialization: unsupported branch Op %T", op)
	}
}

func encodeSequential(ls LayerSpec, op *sequential.Layer) (LayerSpec, error) {
	ls.InputHeight = op.Cfg.Dim
	ls.OutputHeight = op.Cfg.Dim
	ls.DepthN = op.Cfg.Depth
	ls.SeqLength = op.Cfg.SeqLen
	for i, ch := range op.Children {
		if err := addDenseStore(&ls, fmt.Sprintf("child.%d", i), ch); err != nil {
			return ls, err
		}
	}
	return ls, nil
}

func encodeResidual(ls LayerSpec, op *residual.Layer) (LayerSpec, error) {
	ls.InputHeight = op.Cfg.Dim
	ls.OutputHeight = op.Cfg.Dim
	ls.DepthN = op.Cfg.Depth
	ls.SeqLength = op.Cfg.SeqLen
	for i, ch := range op.Children {
		if err := addDenseStore(&ls, fmt.Sprintf("child.%d", i), ch); err != nil {
			return ls, err
		}
	}
	return ls, nil
}

func encodeMeta(ls LayerSpec, op *metacognition.Layer) (LayerSpec, error) {
	ls.InputHeight = op.Cfg.Dim
	ls.OutputHeight = op.Cfg.Dim
	ls.SeqLength = op.Cfg.SeqLen
	return ls, addDenseStore(&ls, "observed", op.Observed)
}

func encodeMamba(ls LayerSpec, op *mamba.Layer) (LayerSpec, error) {
	ls.DModel = op.Cfg.DModel
	ls.DState = op.Cfg.DState
	ls.Expand = op.Cfg.Expand
	ls.SeqLength = op.Cfg.SeqLen
	ls.InputHeight = op.Cfg.DModel
	ls.OutputHeight = op.Cfg.DModel
	if err := addDenseStore(&ls, "in", op.InProj); err != nil {
		return ls, err
	}
	if err := addDenseStore(&ls, "out", op.OutProj); err != nil {
		return ls, err
	}
	if err := putExtra(&ls, "a_log", op.ALog); err != nil {
		return ls, err
	}
	return ls, putExtra(&ls, "d_skip", op.DSkip)
}

func encodeGDN(ls LayerSpec, op *gdn.Layer) (LayerSpec, error) {
	ls.HiddenSize = op.Cfg.HiddenSize
	ls.NumKeyHeads = op.Cfg.NumKeyHeads
	ls.NumValueHeads = op.Cfg.NumValueHeads
	ls.KeyHeadDim = op.Cfg.KeyHeadDim
	ls.ValueHeadDim = op.Cfg.ValueHeadDim
	ls.ConvKernel = op.Cfg.ConvKernel
	ls.Eps = op.Cfg.Eps
	ls.InputHeight = op.Cfg.HiddenSize
	ls.OutputHeight = op.Cfg.HiddenSize
	list := []struct {
		n string
		encode func() (StoreBlob, error)
		ok     bool
	}{
		{"inqkv", func() (StoreBlob, error) { return encodeQuantBlob("inqkv", op.InQKV) }, op.InQKV != nil},
		{"inz", func() (StoreBlob, error) { return encodeQuantBlob("inz", op.InZ) }, op.InZ != nil},
		{"inb", func() (StoreBlob, error) { return encodeQuantBlob("inb", op.InB) }, op.InB != nil},
		{"ina", func() (StoreBlob, error) { return encodeQuantBlob("ina", op.InA) }, op.InA != nil},
		{"out", func() (StoreBlob, error) { return encodeQuantBlob("out", op.Out) }, op.Out != nil},
	}
	for _, it := range list {
		if !it.ok {
			continue
		}
		b, err := it.encode()
		if err != nil {
			return ls, err
		}
		ls.Stores = append(ls.Stores, b)
	}
	if err := putExtra(&ls, "conv", op.ConvWeight); err != nil {
		return ls, err
	}
	if err := putExtra(&ls, "a_log", op.ALog); err != nil {
		return ls, err
	}
	if err := putExtra(&ls, "dt_bias", op.DtBias); err != nil {
		return ls, err
	}
	if err := putExtra(&ls, "norm_gamma", op.NormGamma); err != nil {
		return ls, err
	}
	return ls, nil
}
