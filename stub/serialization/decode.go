package serialization

import (
	"encoding/base64"
	"fmt"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
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
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/weights"
)

func placeFromSpec(g *architecture.Grid, ls LayerSpec) error {
	act := core.ParseActivation(ls.Activation)
	sm := storeMap(ls.Stores)

	// Legacy Dense v0: single Weights field → synthesize "w" store.
	if len(ls.Stores) == 0 && ls.Weights != "" && (ls.Type == "Dense" || ls.Type == "0") {
		raw, err := base64.StdEncoding.DecodeString(ls.Weights)
		if err != nil {
			return err
		}
		w, err := weights.Restore(weights.Snapshot{
			DType:  core.DTypeFloat32,
			Format: quant.FormatNone,
			Rows:   ls.OutputHeight,
			Cols:   ls.InputHeight,
			Scale:  ls.Scale,
			Raw:    raw,
		})
		if err != nil {
			return err
		}
		l, err := denseFromStore(ls.InputHeight, ls.OutputHeight, act, w)
		if err != nil {
			return err
		}
		return dense.Place(g, ls.Z, ls.Y, ls.X, ls.L, l)
	}

	switch ls.Type {
	case "Dense", "0":
		s, err := mustStore(sm, "w")
		if err != nil {
			return err
		}
		l, err := denseFromStore(ls.InputHeight, ls.OutputHeight, act, s)
		if err != nil {
			return err
		}
		return dense.Place(g, ls.Z, ls.Y, ls.X, ls.L, l)

	case "MultiHeadAttention", "1":
		return placeMHA(g, ls, sm)
	case "SwiGLU", "2":
		return placeSwiGLU(g, ls, sm)
	case "RMSNorm", "3":
		return placeRMSNorm(g, ls, sm)
	case "CNN1", "4":
		return placeCNN1(g, ls, sm, act)
	case "CNN2", "5":
		return placeCNN2(g, ls, sm, act)
	case "CNN3", "6":
		return placeCNN3(g, ls, sm, act)
	case "RNN", "7":
		return placeRNN(g, ls, sm)
	case "LSTM", "8":
		return placeLSTM(g, ls, sm)
	case "LayerNorm", "9":
		return placeLayerNorm(g, ls, sm)
	case "ConvTransposed1D", "10":
		return placeConvT1(g, ls, sm, act)
	case "ConvTransposed2D", "11":
		return placeConvT2(g, ls, sm, act)
	case "ConvTransposed3D", "12":
		return placeConvT3(g, ls, sm, act)
	case "Embedding", "13":
		return placeEmbedding(g, ls, sm)
	case "KMeans", "14":
		return placeKMeans(g, ls, sm, act)
	case "Softmax", "15":
		return placeSoftmax(g, ls)
	case "Parallel", "16":
		return placeParallel(g, ls, sm)
	case "Sequential", "17":
		return placeSequential(g, ls, sm)
	case "Residual", "18":
		return placeResidual(g, ls, sm)
	case "Metacognition", "19":
		return placeMeta(g, ls, sm)
	case "Mamba", "20":
		return placeMamba(g, ls, sm)
	case "GDN", "21":
		return placeGDN(g, ls, sm)
	default:
		return fmt.Errorf("serialization: unsupported type %q", ls.Type)
	}
}

func replaceDenseWeights(d *dense.Layer, s *weights.Store) {
	if d == nil || s == nil {
		return
	}
	d.Weights = s
	d.Core.DType = s.DType
}

func placeMHA(g *architecture.Grid, ls LayerSpec, sm map[string]StoreBlob) error {
	cfg := mha.Config{
		DModel:     ls.DModel,
		NumHeads:   ls.NumHeads,
		NumKVHeads: ls.NumKVHeads,
		HeadDim:    ls.HeadDim,
		MaxSeqLen:  ls.SeqLength,
		Causal:     ls.Causal,
		Mask:       parseMask(ls.Mask),
	}
	l, err := mha.New(cfg)
	if err != nil {
		return err
	}
	for _, name := range []string{"q", "k", "v", "o"} {
		s, err := mustStore(sm, name)
		if err != nil {
			return err
		}
		switch name {
		case "q":
			replaceDenseWeights(l.Q, s)
		case "k":
			replaceDenseWeights(l.K, s)
		case "v":
			replaceDenseWeights(l.V, s)
		case "o":
			replaceDenseWeights(l.O, s)
		}
	}
	if qn, err := decodeF32Extra(ls.Extras, "q_norm"); err != nil {
		return err
	} else if len(qn) > 0 {
		l.QNormWeight = f32ToF64(qn)
	}
	if kn, err := decodeF32Extra(ls.Extras, "k_norm"); err != nil {
		return err
	} else if len(kn) > 0 {
		l.KNormWeight = f32ToF64(kn)
	}
	l.Core.DType = core.ParseDType(ls.DType)
	return mha.Place(g, ls.Z, ls.Y, ls.X, ls.L, l)
}

func placeSwiGLU(g *architecture.Grid, ls LayerSpec, sm map[string]StoreBlob) error {
	cfg := swiglu.Config{InputDim: ls.InputHeight, IntermediateDim: ls.IntermediateDim}
	l, err := swiglu.New(cfg)
	if err != nil {
		return err
	}
	for _, pair := range []struct {
		n string
		d **dense.Layer
	}{
		{"gate", &l.Gate}, {"up", &l.Up}, {"down", &l.Down},
	} {
		s, err := mustStore(sm, pair.n)
		if err != nil {
			return err
		}
		replaceDenseWeights(*pair.d, s)
	}
	l.Core.DType = core.ParseDType(ls.DType)
	return swiglu.Place(g, ls.Z, ls.Y, ls.X, ls.L, l)
}

func placeRMSNorm(g *architecture.Grid, ls LayerSpec, sm map[string]StoreBlob) error {
	cfg := rmsnorm.Config{Dim: ls.InputHeight, Eps: ls.Eps}
	l, err := rmsnorm.New(cfg)
	if err != nil {
		return err
	}
	s, err := mustStore(sm, "gamma")
	if err != nil {
		return err
	}
	l.Gamma = s
	l.Core.DType = s.DType
	return rmsnorm.Place(g, ls.Z, ls.Y, ls.X, ls.L, l)
}

func placeLayerNorm(g *architecture.Grid, ls LayerSpec, sm map[string]StoreBlob) error {
	cfg := layernorm.Config{Dim: ls.InputHeight, Eps: ls.Eps}
	l, err := layernorm.New(cfg)
	if err != nil {
		return err
	}
	gma, err := mustStore(sm, "gamma")
	if err != nil {
		return err
	}
	beta, err := mustStore(sm, "beta")
	if err != nil {
		return err
	}
	l.Gamma, l.Beta = gma, beta
	l.Core.DType = gma.DType
	return layernorm.Place(g, ls.Z, ls.Y, ls.X, ls.L, l)
}

func placeCNN1(g *architecture.Grid, ls LayerSpec, sm map[string]StoreBlob, act core.ActivationType) error {
	cfg := cnn1.Config{
		InChannels: ls.InputChannels, Filters: ls.Filters, SeqLen: ls.SeqLength,
		Kernel: ls.KernelSize, Stride: ls.Stride, Padding: ls.Padding, Activation: act,
	}
	l, err := cnn1.New(cfg)
	if err != nil {
		return err
	}
	s, err := mustStore(sm, "proj")
	if err != nil {
		return err
	}
	replaceDenseWeights(l.Proj, s)
	l.Core.DType = s.DType
	return cnn1.Place(g, ls.Z, ls.Y, ls.X, ls.L, l)
}

func placeCNN2(g *architecture.Grid, ls LayerSpec, sm map[string]StoreBlob, act core.ActivationType) error {
	cfg := cnn2.Config{
		InChannels: ls.InputChannels, Filters: ls.Filters,
		Height: ls.InputHeight, Width: ls.InputWidth,
		Kernel: ls.KernelSize, Stride: ls.Stride, Padding: ls.Padding, Activation: act,
	}
	l, err := cnn2.New(cfg)
	if err != nil {
		return err
	}
	s, err := mustStore(sm, "proj")
	if err != nil {
		return err
	}
	replaceDenseWeights(l.Proj, s)
	l.Core.DType = s.DType
	return cnn2.Place(g, ls.Z, ls.Y, ls.X, ls.L, l)
}

func placeCNN3(g *architecture.Grid, ls LayerSpec, sm map[string]StoreBlob, act core.ActivationType) error {
	cfg := cnn3.Config{
		InChannels: ls.InputChannels, Filters: ls.Filters,
		Depth: ls.InputDepth, Height: ls.InputHeight, Width: ls.InputWidth,
		Kernel: ls.KernelSize, Stride: ls.Stride, Padding: ls.Padding, Activation: act,
	}
	l, err := cnn3.New(cfg)
	if err != nil {
		return err
	}
	s, err := mustStore(sm, "proj")
	if err != nil {
		return err
	}
	replaceDenseWeights(l.Proj, s)
	l.Core.DType = s.DType
	return cnn3.Place(g, ls.Z, ls.Y, ls.X, ls.L, l)
}

func placeConvT1(g *architecture.Grid, ls LayerSpec, sm map[string]StoreBlob, act core.ActivationType) error {
	cfg := convt1.Config{
		InChannels: ls.InputChannels, Filters: ls.Filters, SeqLen: ls.SeqLength,
		Kernel: ls.KernelSize, Stride: ls.Stride, Padding: ls.Padding,
		OutputPadding: ls.OutputPadding, Activation: act,
	}
	l, err := convt1.New(cfg)
	if err != nil {
		return err
	}
	s, err := mustStore(sm, "proj")
	if err != nil {
		return err
	}
	replaceDenseWeights(l.Proj, s)
	l.Core.DType = s.DType
	return convt1.Place(g, ls.Z, ls.Y, ls.X, ls.L, l)
}

func placeConvT2(g *architecture.Grid, ls LayerSpec, sm map[string]StoreBlob, act core.ActivationType) error {
	cfg := convt2.Config{
		InChannels: ls.InputChannels, Filters: ls.Filters,
		Height: ls.InputHeight, Width: ls.InputWidth,
		Kernel: ls.KernelSize, Stride: ls.Stride, Padding: ls.Padding,
		OutputPadding: ls.OutputPadding, Activation: act,
	}
	l, err := convt2.New(cfg)
	if err != nil {
		return err
	}
	s, err := mustStore(sm, "proj")
	if err != nil {
		return err
	}
	replaceDenseWeights(l.Proj, s)
	l.Core.DType = s.DType
	return convt2.Place(g, ls.Z, ls.Y, ls.X, ls.L, l)
}

func placeConvT3(g *architecture.Grid, ls LayerSpec, sm map[string]StoreBlob, act core.ActivationType) error {
	cfg := convt3.Config{
		InChannels: ls.InputChannels, Filters: ls.Filters,
		Depth: ls.InputDepth, Height: ls.InputHeight, Width: ls.InputWidth,
		Kernel: ls.KernelSize, Stride: ls.Stride, Padding: ls.Padding,
		OutputPadding: ls.OutputPadding, Activation: act,
	}
	l, err := convt3.New(cfg)
	if err != nil {
		return err
	}
	s, err := mustStore(sm, "proj")
	if err != nil {
		return err
	}
	replaceDenseWeights(l.Proj, s)
	l.Core.DType = s.DType
	return convt3.Place(g, ls.Z, ls.Y, ls.X, ls.L, l)
}

func placeRNN(g *architecture.Grid, ls LayerSpec, sm map[string]StoreBlob) error {
	cfg := rnn.Config{InputSize: ls.InputHeight, HiddenSize: ls.OutputHeight, SeqLen: ls.SeqLength}
	l, err := rnn.New(cfg)
	if err != nil {
		return err
	}
	ih, err := mustStore(sm, "ih")
	if err != nil {
		return err
	}
	hh, err := mustStore(sm, "hh")
	if err != nil {
		return err
	}
	replaceDenseWeights(l.IH, ih)
	replaceDenseWeights(l.HH, hh)
	l.Core.DType = ih.DType
	return rnn.Place(g, ls.Z, ls.Y, ls.X, ls.L, l)
}

func placeLSTM(g *architecture.Grid, ls LayerSpec, sm map[string]StoreBlob) error {
	cfg := lstm.Config{InputSize: ls.InputHeight, HiddenSize: ls.OutputHeight, SeqLen: ls.SeqLength}
	l, err := lstm.New(cfg)
	if err != nil {
		return err
	}
	applyGate := func(g *lstm.Gate, prefix string) error {
		ih, err := mustStore(sm, prefix+"_ih")
		if err != nil {
			return err
		}
		hh, err := mustStore(sm, prefix+"_hh")
		if err != nil {
			return err
		}
		replaceDenseWeights(g.IH, ih)
		replaceDenseWeights(g.HH, hh)
		return nil
	}
	for _, pair := range []struct {
		p string
		g *lstm.Gate
	}{
		{"i", l.I}, {"f", l.F}, {"g", l.G}, {"o", l.O},
	} {
		if err := applyGate(pair.g, pair.p); err != nil {
			return err
		}
	}
	if s, ok := sm["i_ih"]; ok {
		l.Core.DType = core.ParseDType(s.DType)
	}
	return lstm.Place(g, ls.Z, ls.Y, ls.X, ls.L, l)
}

func placeEmbedding(g *architecture.Grid, ls LayerSpec, sm map[string]StoreBlob) error {
	cfg := embedding.Config{VocabSize: ls.VocabSize, EmbeddingDim: ls.EmbeddingDim, SeqLen: ls.SeqLength}
	if cfg.SeqLen <= 0 {
		cfg.SeqLen = 1
	}
	l, err := embedding.New(cfg)
	if err != nil {
		return err
	}
	s, err := mustStore(sm, "w")
	if err != nil {
		return err
	}
	l.Weights = s
	l.Core.DType = s.DType
	return embedding.Place(g, ls.Z, ls.Y, ls.X, ls.L, l)
}

func placeSoftmax(g *architecture.Grid, ls LayerSpec) error {
	cfg := softmax.Config{
		Dim: ls.InputHeight, SeqLen: ls.SeqLength,
		Rows: ls.SoftmaxRows, Cols: ls.SoftmaxCols,
		Temperature: ls.Temperature, Kind: parseSoftmaxKind(ls.SoftmaxKind),
	}
	l, err := softmax.New(cfg)
	if err != nil {
		return err
	}
	l.Core.DType = core.ParseDType(ls.DType)
	return softmax.Place(g, ls.Z, ls.Y, ls.X, ls.L, l)
}

func placeKMeans(g *architecture.Grid, ls LayerSpec, sm map[string]StoreBlob, act core.ActivationType) error {
	cfg := kmeans.Config{
		NumClusters: ls.NumClusters, FeatureDim: ls.InputHeight,
		Temperature: ls.Temperature, OutputMode: kmeans.OutputMode(ls.OutputMode),
		Activation: act,
	}
	l, err := kmeans.New(cfg)
	if err != nil {
		return err
	}
	s, err := mustStore(sm, "centers")
	if err != nil {
		return err
	}
	replaceDenseWeights(l.Centers, s)
	l.Core.DType = s.DType
	return kmeans.Place(g, ls.Z, ls.Y, ls.X, ls.L, l)
}

func placeParallel(g *architecture.Grid, ls LayerSpec, sm map[string]StoreBlob) error {
	cfg := parallel.Config{
		Dim: ls.InputHeight, OutFeat: ls.OutFeat, Branches: ls.Branches,
		Combine: parallel.CombineMode(ls.CombineMode), SeqLen: ls.SeqLength,
	}
	l, err := parallel.New(cfg)
	if err != nil {
		return err
	}
	for i := 0; i < cfg.Branches; i++ {
		s, err := mustStore(sm, fmt.Sprintf("branch.%d", i))
		if err != nil {
			return err
		}
		replaceDenseWeights(l.Branches[i], s)
	}
	if cfg.Combine == parallel.CombineFilter {
		s, err := mustStore(sm, "gate")
		if err != nil {
			return err
		}
		replaceDenseWeights(l.Gate, s)
	}
	if len(l.Branches) > 0 && l.Branches[0].Weights != nil {
		l.Core.DType = l.Branches[0].Weights.DType
	}
	return parallel.Place(g, ls.Z, ls.Y, ls.X, ls.L, l)
}

func placeSequential(g *architecture.Grid, ls LayerSpec, sm map[string]StoreBlob) error {
	cfg := sequential.Config{Dim: ls.InputHeight, SeqLen: ls.SeqLength, Depth: ls.DepthN}
	l, err := sequential.New(cfg)
	if err != nil {
		return err
	}
	for i := 0; i < len(l.Children); i++ {
		s, err := mustStore(sm, fmt.Sprintf("child.%d", i))
		if err != nil {
			return err
		}
		replaceDenseWeights(l.Children[i], s)
	}
	if len(l.Children) > 0 && l.Children[0].Weights != nil {
		l.Core.DType = l.Children[0].Weights.DType
	}
	return sequential.Place(g, ls.Z, ls.Y, ls.X, ls.L, l)
}

func placeResidual(g *architecture.Grid, ls LayerSpec, sm map[string]StoreBlob) error {
	cfg := residual.Config{Dim: ls.InputHeight, SeqLen: ls.SeqLength, Depth: ls.DepthN}
	l, err := residual.New(cfg)
	if err != nil {
		return err
	}
	for i := 0; i < len(l.Children); i++ {
		s, err := mustStore(sm, fmt.Sprintf("child.%d", i))
		if err != nil {
			return err
		}
		replaceDenseWeights(l.Children[i], s)
	}
	if len(l.Children) > 0 && l.Children[0].Weights != nil {
		l.Core.DType = l.Children[0].Weights.DType
	}
	return residual.Place(g, ls.Z, ls.Y, ls.X, ls.L, l)
}

func placeMeta(g *architecture.Grid, ls LayerSpec, sm map[string]StoreBlob) error {
	cfg := metacognition.Config{Dim: ls.InputHeight, SeqLen: ls.SeqLength}
	l, err := metacognition.New(cfg)
	if err != nil {
		return err
	}
	s, err := mustStore(sm, "observed")
	if err != nil {
		return err
	}
	replaceDenseWeights(l.Observed, s)
	l.Core.DType = s.DType
	return metacognition.Place(g, ls.Z, ls.Y, ls.X, ls.L, l)
}

func placeMamba(g *architecture.Grid, ls LayerSpec, sm map[string]StoreBlob) error {
	cfg := mamba.Config{DModel: ls.DModel, DState: ls.DState, Expand: ls.Expand, SeqLen: ls.SeqLength}
	aLog, err := decodeF32Extra(ls.Extras, "a_log")
	if err != nil {
		return err
	}
	dSkip, err := decodeF32Extra(ls.Extras, "d_skip")
	if err != nil {
		return err
	}
	l, err := mamba.NewConfigured[float32](cfg, core.DTypeFloat32, quant.FormatNone, nil, nil, aLog, dSkip)
	if err != nil {
		return err
	}
	inS, err := mustStore(sm, "in")
	if err != nil {
		return err
	}
	outS, err := mustStore(sm, "out")
	if err != nil {
		return err
	}
	replaceDenseWeights(l.InProj, inS)
	replaceDenseWeights(l.OutProj, outS)
	l.Core.DType = inS.DType
	return mamba.Place(g, ls.Z, ls.Y, ls.X, ls.L, l)
}

func placeGDN(g *architecture.Grid, ls LayerSpec, sm map[string]StoreBlob) error {
	cfg := gdn.Config{
		HiddenSize: ls.HiddenSize, NumKeyHeads: ls.NumKeyHeads, NumValueHeads: ls.NumValueHeads,
		KeyHeadDim: ls.KeyHeadDim, ValueHeadDim: ls.ValueHeadDim, ConvKernel: ls.ConvKernel, Eps: ls.Eps,
	}
	l, err := gdn.New(cfg)
	if err != nil {
		return err
	}
	set := func(name string, dst **quant.Blob) error {
		b, ok := sm[name]
		if !ok {
			return fmt.Errorf("serialization: missing gdn store %q", name)
		}
		qb, err := decodeQuantBlob(b)
		if err != nil {
			return err
		}
		*dst = qb
		return nil
	}
	if err := set("inqkv", &l.InQKV); err != nil {
		return err
	}
	if err := set("inz", &l.InZ); err != nil {
		return err
	}
	if err := set("inb", &l.InB); err != nil {
		return err
	}
	if err := set("ina", &l.InA); err != nil {
		return err
	}
	if err := set("out", &l.Out); err != nil {
		return err
	}
	if v, err := decodeF32Extra(ls.Extras, "conv"); err != nil {
		return err
	} else if v != nil {
		l.ConvWeight = v
	}
	if v, err := decodeF32Extra(ls.Extras, "a_log"); err != nil {
		return err
	} else if v != nil {
		l.ALog = v
	}
	if v, err := decodeF32Extra(ls.Extras, "dt_bias"); err != nil {
		return err
	} else if v != nil {
		l.DtBias = v
	}
	if v, err := decodeF32Extra(ls.Extras, "norm_gamma"); err != nil {
		return err
	} else if v != nil {
		l.NormGamma = v
	}
	return gdn.Place(g, ls.Z, ls.Y, ls.X, ls.L, l)
}

func parseMask(s string) mha.MaskKind {
	switch s {
	case "causal":
		return mha.MaskCausal
	case "bidirectional":
		return mha.MaskBidirectional
	case "sliding_window":
		return mha.MaskSlidingWindow
	case "prefix_lm":
		return mha.MaskPrefixLM
	case "custom":
		return mha.MaskCustom
	default:
		return mha.MaskUnspecified
	}
}

func parseSoftmaxKind(s string) softmax.Kind {
	switch s {
	case "grid":
		return softmax.KindGrid
	default:
		return softmax.KindStandard
	}
}

func f32ToF64(v []float32) []float64 {
	out := make([]float64, len(v))
	for i := range v {
		out[i] = float64(v[i])
	}
	return out
}
