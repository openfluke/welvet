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
	"github.com/openfluke/welvet/weights"
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

	if !spec.LMHeadTied {
		lm, err := ef.LoadBlob("transformer.lm_head")
		if err != nil {
			return nil, err
		}
		ws, err := weights.New(spec.VocabSize, spec.HiddenSize, lm, core.DTypeFloat32, quant.FormatNone)
		if err != nil {
			return nil, fmt.Errorf("lm_head: %w", err)
		}
		m.lmHead = ws
	}

	for i := 0; i < d.NumLayers; i++ {
		prefix := fmt.Sprintf("blocks.%d", i)
		load := func(suffix string) ([]float32, error) {
			return ef.LoadBlob(prefix + "." + suffix)
		}
		an, err := load("attn_norm")
		if err != nil {
			return nil, err
		}
		attnNorm, err := rmsnorm.NewConfigured(rmsnorm.Config{Dim: spec.HiddenSize, Eps: eps}, core.DTypeFloat32, quant.FormatNone, an)
		if err != nil {
			return nil, err
		}
		q, err := load("q")
		if err != nil {
			return nil, err
		}
		k, err := load("k")
		if err != nil {
			return nil, err
		}
		v, err := load("v")
		if err != nil {
			return nil, err
		}
		o, err := load("o")
		if err != nil {
			return nil, err
		}
		cfg := mha.DecoderCausal(spec.HiddenSize, d.NumHeads, d.NumKVHeads)
		cfg.HeadDim = d.HeadDim
		cfg.RoPETheta = rope
		cfg.MaxSeqLen = maxSeq
		attn, err := mha.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone, q, k, v, o)
		if err != nil {
			return nil, fmt.Errorf("block %d mha: %w", i, err)
		}

		fn, err := load("ffn_norm")
		if err != nil {
			return nil, err
		}
		ffnNorm, err := rmsnorm.NewConfigured(rmsnorm.Config{Dim: spec.HiddenSize, Eps: eps}, core.DTypeFloat32, quant.FormatNone, fn)
		if err != nil {
			return nil, err
		}
		gate, err := load("gate")
		if err != nil {
			return nil, err
		}
		up, err := load("up")
		if err != nil {
			return nil, err
		}
		down, err := load("down")
		if err != nil {
			return nil, err
		}
		ffn, err := swiglu.NewConfigured(swiglu.Config{
			InputDim:         spec.HiddenSize,
			IntermediateDim:  d.IntermediateSize,
		}, core.DTypeFloat32, quant.FormatNone, gate, up, down)
		if err != nil {
			return nil, fmt.Errorf("block %d swiglu: %w", i, err)
		}
		m.Blocks[i] = Block{AttnNorm: attnNorm, Attn: attn, FFNNorm: ffnNorm, FFN: ffn}
	}

	return m, nil
}

// ResetKV clears attention caches for a new prompt.
func (m *Model) ResetKV() {
	if m == nil {
		return
	}
	for i := range m.Blocks {
		m.Blocks[i].Attn.KVOffset = 0
		m.Blocks[i].Attn.KVCacheK = nil
		m.Blocks[i].Attn.KVCacheV = nil
	}
}
