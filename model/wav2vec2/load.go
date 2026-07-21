package wav2vec2

import (
	"fmt"
	"math"
	"path/filepath"

	"github.com/openfluke/welvet/model/hf"
)

// LoadHFDir loads facebook/wav2vec2-base-960h from a HF snapshot directory
// containing config.json, vocab.json, and model.safetensors.
func LoadHFDir(dir string) (*Model, error) {
	cfg, err := LoadConfigJSON(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	vocab, err := LoadVocabJSON(filepath.Join(dir, "vocab.json"), cfg.PadTokenID)
	if err != nil {
		return nil, err
	}
	tensors, err := hf.LoadSafetensorsWithMeta(filepath.Join(dir, "model.safetensors"), nil)
	if err != nil {
		return nil, err
	}
	return BuildFromTensors(cfg, vocab, tensors)
}

// BuildFromTensors wires a Model from decoded safetensors.
func BuildFromTensors(cfg Config, vocab *Vocab, tensors map[string]hf.TensorWithMeta) (*Model, error) {
	must := func(name string) ([]float32, []int, error) {
		t, ok := tensors[name]
		if !ok {
			return nil, nil, fmt.Errorf("wav2vec2: missing tensor %s", name)
		}
		return t.Data, t.Shape, nil
	}
	m := &Model{Cfg: cfg, Vocab: vocab}

	nFeats := len(cfg.ConvDim)
	m.Feats = make([]convLayer, nFeats)
	inC := 1
	for i := 0; i < nFeats; i++ {
		name := fmt.Sprintf("wav2vec2.feature_extractor.conv_layers.%d.conv.weight", i)
		w, shape, err := must(name)
		if err != nil {
			return nil, err
		}
		if len(shape) != 3 {
			return nil, fmt.Errorf("wav2vec2: %s shape %v", name, shape)
		}
		cl := convLayer{
			W:      w,
			OutC:   shape[0],
			InC:    shape[1],
			K:      shape[2],
			Stride: cfg.ConvStride[i],
		}
		if i == 0 && cfg.FeatExtractNorm == "group" {
			nw, _, err := must("wav2vec2.feature_extractor.conv_layers.0.layer_norm.weight")
			if err != nil {
				return nil, err
			}
			nb, _, err := must("wav2vec2.feature_extractor.conv_layers.0.layer_norm.bias")
			if err != nil {
				return nil, err
			}
			cl.NormW, cl.NormB, cl.HasGroupNorm = nw, nb, true
		}
		if cl.OutC != cfg.ConvDim[i] || cl.InC != inC || cl.K != cfg.ConvKernel[i] {
			return nil, fmt.Errorf("wav2vec2: conv %d shape mismatch got %dx%dx%d", i, cl.OutC, cl.InC, cl.K)
		}
		m.Feats[i] = cl
		inC = cl.OutC
	}

	pnw, _, err := must("wav2vec2.feature_projection.layer_norm.weight")
	if err != nil {
		return nil, err
	}
	pnb, _, err := must("wav2vec2.feature_projection.layer_norm.bias")
	if err != nil {
		return nil, err
	}
	m.ProjNorm = layerNormParams{W: pnw, B: pnb}
	pw, pshape, err := must("wav2vec2.feature_projection.projection.weight")
	if err != nil {
		return nil, err
	}
	pb, _, err := must("wav2vec2.feature_projection.projection.bias")
	if err != nil {
		return nil, err
	}
	m.Proj = linearLayer{W: pw, B: pb, Out: pshape[0], In: pshape[1]}

	posW, err := fusePosConvWeight(tensors)
	if err != nil {
		return nil, err
	}
	posB, _, err := must("wav2vec2.encoder.pos_conv_embed.conv.bias")
	if err != nil {
		return nil, err
	}
	m.PosW = posW
	m.PosB = posB
	m.PosK = cfg.NumConvPosEmbeddings
	m.PosGroups = cfg.NumConvPosEmbeddingGroups
	m.PosInPerG = cfg.HiddenSize / cfg.NumConvPosEmbeddingGroups

	enw, _, err := must("wav2vec2.encoder.layer_norm.weight")
	if err != nil {
		return nil, err
	}
	enb, _, err := must("wav2vec2.encoder.layer_norm.bias")
	if err != nil {
		return nil, err
	}
	m.EncNorm = layerNormParams{W: enw, B: enb}

	m.Layers = make([]encoderLayer, cfg.NumHiddenLayers)
	for i := 0; i < cfg.NumHiddenLayers; i++ {
		el, err := loadEncoderLayer(must, i, cfg)
		if err != nil {
			return nil, err
		}
		m.Layers[i] = el
	}

	lw, lshape, err := must("lm_head.weight")
	if err != nil {
		return nil, err
	}
	lb, _, err := must("lm_head.bias")
	if err != nil {
		return nil, err
	}
	m.LMHead = linearLayer{W: lw, B: lb, Out: lshape[0], In: lshape[1]}
	return m, nil
}

func loadEncoderLayer(must func(string) ([]float32, []int, error), i int, cfg Config) (encoderLayer, error) {
	prefix := fmt.Sprintf("wav2vec2.encoder.layers.%d", i)
	loadLin := func(name string) (linearLayer, error) {
		w, shape, err := must(name + ".weight")
		if err != nil {
			return linearLayer{}, err
		}
		b, _, err := must(name + ".bias")
		if err != nil {
			return linearLayer{}, err
		}
		return linearLayer{W: w, B: b, Out: shape[0], In: shape[1]}, nil
	}
	loadLN := func(name string) (layerNormParams, error) {
		w, _, err := must(name + ".weight")
		if err != nil {
			return layerNormParams{}, err
		}
		b, _, err := must(name + ".bias")
		if err != nil {
			return layerNormParams{}, err
		}
		return layerNormParams{W: w, B: b}, nil
	}
	var el encoderLayer
	var err error
	if el.Q, err = loadLin(prefix + ".attention.q_proj"); err != nil {
		return el, err
	}
	if el.K, err = loadLin(prefix + ".attention.k_proj"); err != nil {
		return el, err
	}
	if el.V, err = loadLin(prefix + ".attention.v_proj"); err != nil {
		return el, err
	}
	if el.Out, err = loadLin(prefix + ".attention.out_proj"); err != nil {
		return el, err
	}
	if el.AttnNorm, err = loadLN(prefix + ".layer_norm"); err != nil {
		return el, err
	}
	if el.FFInter, err = loadLin(prefix + ".feed_forward.intermediate_dense"); err != nil {
		return el, err
	}
	if el.FFOut, err = loadLin(prefix + ".feed_forward.output_dense"); err != nil {
		return el, err
	}
	if el.FinalNorm, err = loadLN(prefix + ".final_layer_norm"); err != nil {
		return el, err
	}
	_ = cfg
	return el, nil
}

// fusePosConvWeight applies PyTorch weight_norm(..., dim=2):
// g shape [1,1,K], v [out,in_per_group,K], w = v * g / ||v||_{dims0,1}.
func fusePosConvWeight(tensors map[string]hf.TensorWithMeta) ([]float32, error) {
	gT, ok := tensors["wav2vec2.encoder.pos_conv_embed.conv.weight_g"]
	if !ok {
		return nil, fmt.Errorf("wav2vec2: missing pos conv weight_g")
	}
	vT, ok := tensors["wav2vec2.encoder.pos_conv_embed.conv.weight_v"]
	if !ok {
		return nil, fmt.Errorf("wav2vec2: missing pos conv weight_v")
	}
	v, g := vT.Data, gT.Data
	if len(vT.Shape) != 3 {
		return nil, fmt.Errorf("wav2vec2: pos weight_v shape %v", vT.Shape)
	}
	outC, inG, k := vT.Shape[0], vT.Shape[1], vT.Shape[2]
	if len(g) != k {
		return nil, fmt.Errorf("wav2vec2: pos weight_g len %d want %d", len(g), k)
	}
	w := make([]float32, len(v))
	for ki := 0; ki < k; ki++ {
		var sum float64
		for oc := 0; oc < outC; oc++ {
			for ic := 0; ic < inG; ic++ {
				x := float64(v[(oc*inG+ic)*k+ki])
				sum += x * x
			}
		}
		inv := float64(g[ki]) / math.Sqrt(sum)
		for oc := 0; oc < outC; oc++ {
			for ic := 0; ic < inG; ic++ {
				idx := (oc*inG+ic)*k + ki
				w[idx] = float32(float64(v[idx]) * inv)
			}
		}
	}
	return w, nil
}
