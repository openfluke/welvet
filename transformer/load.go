package transformer

import (
	"fmt"
	"path/filepath"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/embedding"
	"github.com/openfluke/welvet/entity"
	"github.com/openfluke/welvet/hf"
	"github.com/openfluke/welvet/mha"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/rmsnorm"
	"github.com/openfluke/welvet/swiglu"
)

// LoadEntity builds a Model from a packed Welvet .entity file.
func LoadEntity(path string) (*Model, error) {
	ef, err := entity.Open(path)
	if err != nil {
		return nil, err
	}
	defer ef.Close()

	hdr := ef.Header()
	if hdr == nil || hdr.Transformer == nil || hdr.Transformer.Dims == nil {
		return nil, fmt.Errorf("entity: missing transformer dims")
	}
	if hdr.Status != "" && hdr.Status != "packed" {
		return nil, fmt.Errorf("entity status=%q (need packed)", hdr.Status)
	}
	spec := hdr.Transformer
	d := spec.Dims
	maxSeq := 2048
	if d.NumLayers <= 0 || d.NumHeads <= 0 || spec.HiddenSize <= 0 {
		return nil, fmt.Errorf("entity: invalid dims")
	}

	// Qwen3.5 hybrid / dense Qwen3 BinaryPacked (Bonsai) path
	if spec.Architecture == "qwen35_hybrid" || spec.Architecture == "qwen3_dense" || len(d.LayerTypes) > 0 {
		m := &Model{
			HiddenSize:    spec.HiddenSize,
			VocabSize:     spec.VocabSize,
			LMHeadTied:    spec.LMHeadTied,
			HasFinalNorm:  spec.HasFinalNorm,
			MaxSeqLen:     maxSeq,
			Repo:          spec.Repo,
			Snapshot:      spec.Snapshot,
			TokenizerPath: spec.Tokenizer,
			Blocks:        make([]Block, d.NumLayers),
			EOSTokens:     []int{248046},
		}
		if m.Snapshot != "" {
			cfgPath := filepath.Join(m.Snapshot, "config.json")
			if cfg, err := hf.LoadConfigJSON(cfgPath); err == nil {
				m.EOSTokens = hf.EOSTokenIDs(cfg)
			}
		}
		if err := loadHybridEntity(ef, m, spec); err != nil {
			return nil, err
		}
		return m, nil
	}

	embData, err := ef.LoadBlob("transformer.embeddings")
	if err != nil {
		return nil, err
	}
	emb, err := embedding.NewConfigured(embedding.Config{
		VocabSize:    spec.VocabSize,
		EmbeddingDim: spec.HiddenSize,
		SeqLen:       maxSeq,
	}, core.DTypeFloat32, quant.FormatNone, embData)
	if err != nil {
		return nil, fmt.Errorf("embed: %w", err)
	}

	m := &Model{
		HiddenSize:    spec.HiddenSize,
		VocabSize:     spec.VocabSize,
		LMHeadTied:    spec.LMHeadTied,
		HasFinalNorm:  spec.HasFinalNorm,
		MaxSeqLen:     maxSeq,
		Repo:          spec.Repo,
		Snapshot:      spec.Snapshot,
		TokenizerPath: spec.Tokenizer,
		Embed:         emb,
		Blocks:        make([]Block, d.NumLayers),
		EOSTokens:     []int{2},
	}
	packFmt := specPackFormat(spec)
	m.PackFormat = packFmt
	m.FusedPack = packFmt != quant.FormatNone
	if m.Snapshot != "" {
		cfgPath := filepath.Join(m.Snapshot, "config.json")
		if cfg, err := hf.LoadConfigJSON(cfgPath); err == nil {
			m.EOSTokens = hf.EOSTokenIDs(cfg)
		}
	}

	eps := d.RMSNormEps
	if eps <= 0 {
		eps = 1e-6
	}
	rope := d.RoPEFreqBase
	if rope <= 0 {
		rope = 10000
	}

	if spec.HasFinalNorm {
		fn, err := ef.LoadBlob("transformer.final_norm")
		if err != nil {
			return nil, err
		}
		m.FinalNorm, err = rmsnorm.NewConfigured(rmsnorm.Config{Dim: spec.HiddenSize, Eps: eps}, core.DTypeFloat32, quant.FormatNone, fn)
		if err != nil {
			return nil, fmt.Errorf("final_norm: %w", err)
		}
	}

	if err := loadLMHead(ef, m, packFmt); err != nil {
		return nil, err
	}

	qDim := d.NumHeads * d.HeadDim
	kvDim := d.NumKVHeads * d.HeadDim
	if d.QueryDim > 0 {
		qDim = d.QueryDim
	}
	if d.KVDim > 0 {
		kvDim = d.KVDim
	}
	hidden := spec.HiddenSize
	inter := d.IntermediateSize

	for i := 0; i < d.NumLayers; i++ {
		prefix := fmt.Sprintf("blocks.%d", i)
		loadF32 := func(suffix string) ([]float32, error) {
			return ef.LoadBlob(prefix + "." + suffix)
		}
		an, err := loadF32("attn_norm")
		if err != nil {
			return nil, err
		}
		attnNorm, err := rmsnorm.NewConfigured(rmsnorm.Config{Dim: spec.HiddenSize, Eps: eps}, core.DTypeFloat32, quant.FormatNone, an)
		if err != nil {
			return nil, err
		}
		qStore, err := loadWeightStore(ef, prefix+".q", qDim, hidden, packFmt)
		if err != nil {
			return nil, err
		}
		kStore, err := loadWeightStore(ef, prefix+".k", kvDim, hidden, packFmt)
		if err != nil {
			return nil, err
		}
		vStore, err := loadWeightStore(ef, prefix+".v", kvDim, hidden, packFmt)
		if err != nil {
			return nil, err
		}
		oStore, err := loadWeightStore(ef, prefix+".o", hidden, qDim, packFmt)
		if err != nil {
			return nil, err
		}
		cfg := mha.DecoderCausal(spec.HiddenSize, d.NumHeads, d.NumKVHeads)
		cfg.HeadDim = d.HeadDim
		cfg.RoPETheta = rope
		cfg.MaxSeqLen = maxSeq
		attn := &mha.Layer{
			Core: core.Layer{
				Type:         core.LayerMultiHeadAttention,
				DType:        core.DTypeFloat32,
				Activation:   core.ActivationLinear,
				InputHeight:  hidden,
				OutputHeight: hidden,
				TileSize:     32,
				MultiCore:    true,
			},
			Cfg:  cfg,
			Exec: core.ExecConfig{Backend: core.BackendCPUTiled, MultiCore: true, TileSize: 32},
			Q:    denseFromStore(hidden, qDim, core.ActivationLinear, qStore),
			K:    denseFromStore(hidden, kvDim, core.ActivationLinear, kStore),
			V:    denseFromStore(hidden, kvDim, core.ActivationLinear, vStore),
			O:    denseFromStore(qDim, hidden, core.ActivationLinear, oStore),
		}

		fn, err := loadF32("ffn_norm")
		if err != nil {
			return nil, err
		}
		ffnNorm, err := rmsnorm.NewConfigured(rmsnorm.Config{Dim: spec.HiddenSize, Eps: eps}, core.DTypeFloat32, quant.FormatNone, fn)
		if err != nil {
			return nil, err
		}
		gateStore, err := loadWeightStore(ef, prefix+".gate", inter, hidden, packFmt)
		if err != nil {
			return nil, err
		}
		upStore, err := loadWeightStore(ef, prefix+".up", inter, hidden, packFmt)
		if err != nil {
			return nil, err
		}
		downStore, err := loadWeightStore(ef, prefix+".down", hidden, inter, packFmt)
		if err != nil {
			return nil, err
		}
		ffn := &swiglu.Layer{
			Core: core.Layer{
				Type:         core.LayerSwiGLU,
				DType:        core.DTypeFloat32,
				Activation:   core.ActivationSilu,
				InputHeight:  hidden,
				OutputHeight: hidden,
				TileSize:     32,
				MultiCore:    true,
			},
			Cfg: swiglu.Config{
				InputDim:        hidden,
				IntermediateDim: inter,
			},
			Exec: core.ExecConfig{Backend: core.BackendCPUTiled, MultiCore: true, TileSize: 32},
			Gate: denseFromStore(hidden, inter, core.ActivationLinear, gateStore),
			Up:   denseFromStore(hidden, inter, core.ActivationLinear, upStore),
			Down: denseFromStore(inter, hidden, core.ActivationLinear, downStore),
		}
		m.Blocks[i] = Block{AttnNorm: attnNorm, Attn: attn, FFNNorm: ffnNorm, FFN: ffn}
	}

	return m, nil
}
