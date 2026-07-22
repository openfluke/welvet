package entity

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/openfluke/welvet/model/hf"
	"github.com/openfluke/welvet/quant"
)

// PackFromQwen3MLX converts a dense Qwen3 / Bonsai-8B MLX 1-bit snapshot to a Welvet .entity.
// Uses BinaryPacked g128 (native MLX layout). No GDN layers — all full_attention.
func PackFromQwen3MLX(snapshotDir, outPath string, opts PackOptions) error {
	configPath := filepath.Join(snapshotDir, "config.json")
	config, err := hf.LoadConfigJSON(configPath)
	if err != nil {
		return fmt.Errorf("config.json: %w", err)
	}
	if hf.IsQwen35Hybrid(config) {
		return fmt.Errorf("snapshot is Qwen3.5 hybrid — use PackFromQwen35MLX")
	}
	bits, group, qok := hf.QuantBitsGroup(config)
	if !qok || bits != 1 || group != 128 {
		return fmt.Errorf("expected MLX 1-bit g128 quantization, got bits=%d group=%d ok=%v", bits, group, qok)
	}

	sts, err := filepath.Glob(filepath.Join(snapshotDir, "*.safetensors"))
	if err != nil {
		return err
	}
	if len(sts) == 0 {
		return fmt.Errorf("no *.safetensors in %s", snapshotDir)
	}
	stPath := sts[0]
	for _, p := range sts {
		base := filepath.Base(p)
		if strings.HasSuffix(base, "model.safetensors") && !strings.Contains(base, "index") {
			stPath = p
			break
		}
	}

	cfg := hf.EffectiveConfig(config)
	dims, err := hf.ParseDecoderDims(config, sts)
	if err != nil {
		return err
	}

	index, err := hf.BuildTensorIndex(stPath)
	if err != nil {
		return err
	}

	// Dense Bonsai uses model.*; some exports may wrap under language_model.model.*.
	modelPrefix := "model"
	if _, ok := index["model.embed_tokens.weight"]; !ok {
		if _, ok := index["language_model.model.embed_tokens.weight"]; ok {
			modelPrefix = "language_model.model"
		} else {
			return fmt.Errorf("no dense embed_tokens in %s", filepath.Base(stPath))
		}
	}
	lmPrefix := "lm_head"
	if _, ok := index["lm_head.weight"]; !ok {
		if _, ok := index["language_model.lm_head.weight"]; ok {
			lmPrefix = "language_model.lm_head"
		}
	}

	layerTypes := hf.LayerTypes(config)
	if len(layerTypes) == 0 {
		layerTypes = make([]string, dims.NumLayers)
		for i := range layerTypes {
			layerTypes[i] = "full_attention"
		}
	}
	if len(layerTypes) != dims.NumLayers {
		return fmt.Errorf("layer_types len %d != num_layers %d", len(layerTypes), dims.NumLayers)
	}
	for i, t := range layerTypes {
		if t != "full_attention" {
			return fmt.Errorf("dense Qwen3 packer: layer %d type %q (only full_attention supported)", i, t)
		}
	}

	progress := opts.Progress
	report := func(i, n int, detail string) {
		if progress != nil {
			progress(i, n, detail)
		}
	}
	report(0, dims.NumLayers, "loading globals…")

	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		return err
	}
	payloadTmp, err := os.CreateTemp(filepath.Dir(outPath), ".entity-payload-*")
	if err != nil {
		return err
	}
	payloadPath := payloadTmp.Name()
	defer os.Remove(payloadPath)
	acc := &payloadAcc{w: payloadTmp}
	var blobs []WeightBlob

	appendF32 := func(path string, data []float32) error {
		off := acc.offset
		if err := acc.writeF32(data); err != nil {
			return err
		}
		blobs = append(blobs, WeightBlob{
			Path: path, Offset: off, Length: uint64(len(data) * 4),
			DType: "FLOAT32", Format: "none", Native: false,
		})
		return nil
	}
	appendBlob := func(path string, b *quant.Blob) error {
		wire := EncodePackedBlob(b)
		off := acc.offset
		if _, err := acc.w.Write(wire); err != nil {
			return err
		}
		acc.offset += uint64(len(wire))
		blobs = append(blobs, WeightBlob{
			Path: path, Offset: off, Length: uint64(len(wire)),
			DType: "PACKED", Format: b.Format.String(),
			Rows: b.Rows, Cols: b.Cols, Native: true,
		})
		return nil
	}

	embBlob, err := hf.LoadMLX1BitMatrix(stPath, index, modelPrefix+".embed_tokens")
	if err != nil {
		return fmt.Errorf("embed: %w", err)
	}
	if err := appendBlob("transformer.embeddings.packed", embBlob); err != nil {
		return err
	}
	dims.VocabSize = embBlob.Rows

	fn, err := hf.LoadF16Vector(stPath, index, modelPrefix+".norm.weight")
	if err != nil {
		return fmt.Errorf("final_norm: %w", err)
	}
	if err := appendF32("transformer.final_norm", fn); err != nil {
		return err
	}

	tied := false
	if v, ok := cfg["tie_word_embeddings"].(bool); ok {
		tied = v
	}
	if _, hasLM := index[lmPrefix+".weight"]; hasLM {
		lmBlob, err := hf.LoadMLX1BitMatrix(stPath, index, lmPrefix)
		if err != nil {
			return fmt.Errorf("lm_head: %w", err)
		}
		if err := appendBlob("transformer.lm_head.packed", lmBlob); err != nil {
			return err
		}
		tied = false
	} else if !tied {
		return fmt.Errorf("missing %s.weight (and tie_word_embeddings=false)", lmPrefix)
	}
	// else tied: LM head reuses embeddings.packed at load time

	for i := 0; i < dims.NumLayers; i++ {
		report(i+1, dims.NumLayers, fmt.Sprintf("layer %d (full_attention)", i))
		prefix := fmt.Sprintf("blocks.%d", i)
		lp := fmt.Sprintf("%s.layers.%d", modelPrefix, i)

		an, err := hf.LoadF16Vector(stPath, index, lp+".input_layernorm.weight")
		if err != nil {
			return err
		}
		if err := appendF32(prefix+".attn_norm", an); err != nil {
			return err
		}
		fnorm, err := hf.LoadF16Vector(stPath, index, lp+".post_attention_layernorm.weight")
		if err != nil {
			return err
		}
		if err := appendF32(prefix+".ffn_norm", fnorm); err != nil {
			return err
		}

		for _, role := range []struct{ hf, ent string }{
			{"mlp.gate_proj", "gate"},
			{"mlp.up_proj", "up"},
			{"mlp.down_proj", "down"},
			{"self_attn.q_proj", "q"},
			{"self_attn.k_proj", "k"},
			{"self_attn.v_proj", "v"},
			{"self_attn.o_proj", "o"},
		} {
			b, err := hf.LoadMLX1BitMatrix(stPath, index, lp+"."+role.hf)
			if err != nil {
				return fmt.Errorf("layer %d %s: %w", i, role.hf, err)
			}
			if err := appendBlob(prefix+"."+role.ent, b); err != nil {
				return err
			}
		}
		qn, err := hf.LoadF16Vector(stPath, index, lp+".self_attn.q_norm.weight")
		if err != nil {
			return fmt.Errorf("layer %d q_norm: %w", i, err)
		}
		if err := appendF32(prefix+".q_norm", qn); err != nil {
			return err
		}
		kn, err := hf.LoadF16Vector(stPath, index, lp+".self_attn.k_norm.weight")
		if err != nil {
			return fmt.Errorf("layer %d k_norm: %w", i, err)
		}
		if err := appendF32(prefix+".k_norm", kn); err != nil {
			return err
		}
	}

	tokPath := appendTokenizerBlob(acc, &blobs, snapshotDir)

	if err := payloadTmp.Close(); err != nil {
		return err
	}

	if tokPath == "" {
		cand := filepath.Join(snapshotDir, "tokenizer.json")
		if _, err := os.Stat(cand); err == nil {
			tokPath = cand
		}
	}

	rope := dims.RoPEFreqBase
	if v, ok := cfg["rope_theta"].(float64); ok && v > 0 {
		rope = v
	}

	spec := &TransformerSpec{
		Architecture: "qwen3_dense",
		HiddenSize:   dims.HiddenSize,
		VocabSize:    dims.VocabSize,
		MaxSeqLen:    hf.MaxSeqLenFromConfig(cfg),
		LMHeadTied:   tied,
		HasFinalNorm: true,
		WeightDType:  "PACKED",
		PackFormat:   quant.FormatBinaryPacked.String(),
		LMHeadPacked: true,
		Snapshot:     snapshotDir,
		Tokenizer:    tokPath,
		Repo:         opts.Repo,
		Engine:       "welvet",
		Dims: &TransformerDims{
			NumLayers:        dims.NumLayers,
			NumHeads:         dims.NumHeads,
			NumKVHeads:       dims.NumKVHeads,
			HeadDim:          dims.HeadDim,
			QueryDim:         dims.NumHeads * dims.HeadDim,
			KVDim:            dims.NumKVHeads * dims.HeadDim,
			IntermediateSize: dims.IntermediateSize,
			RMSNormEps:       dims.RMSNormEps,
			RoPEFreqBase:     rope,
			PartialRotaryFactor: 1.0,
			AttnOutputGate:   false,
			LayerTypes:       layerTypes,
		},
	}

	payload, err := os.ReadFile(payloadPath)
	if err != nil {
		return err
	}
	return WriteTransformerFile(outPath, spec, blobs, payload)
}
