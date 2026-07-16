package entity

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/openfluke/welvet/hf"
	"github.com/openfluke/welvet/quant"
)

// PackFromQwen35MLX converts a Qwen3.5 / Bonsai MLX 1-bit snapshot to a Welvet .entity.
// Vision tower tensors are skipped (text-only). Weights stay BinaryPacked g128 (native MLX layout).
func PackFromQwen35MLX(snapshotDir, outPath string, opts PackOptions) error {
	configPath := filepath.Join(snapshotDir, "config.json")
	config, err := hf.LoadConfigJSON(configPath)
	if err != nil {
		return fmt.Errorf("config.json: %w", err)
	}
	if hf.DetectArchitecture(config) != hf.ArchQwen35Hybrid {
		return fmt.Errorf("not a Qwen3.5 hybrid snapshot (model_type=%q)", hf.ConfigStringDefault(config, "model_type", ""))
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
	// Prefer single merged file; otherwise first file with language_model tensors.
	stPath := sts[0]
	for _, p := range sts {
		if strings.HasSuffix(filepath.Base(p), "model.safetensors") && !strings.Contains(filepath.Base(p), "index") {
			stPath = p
			break
		}
	}

	cfg := hf.EffectiveConfig(config)
	dims, err := hf.ParseDecoderDims(config, sts)
	if err != nil {
		return err
	}
	lin := hf.ParseLinearAttnDims(config)
	if err := hf.ValidateHybridDims(lin); err != nil {
		return err
	}
	layerTypes := hf.LayerTypes(config)
	if len(layerTypes) != dims.NumLayers {
		return fmt.Errorf("layer_types len %d != num_layers %d", len(layerTypes), dims.NumLayers)
	}

	index, err := hf.BuildTensorIndex(stPath)
	if err != nil {
		return err
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

	// Embeddings (packed binary g128)
	embBlob, err := hf.LoadMLX1BitMatrix(stPath, index, "language_model.model.embed_tokens")
	if err != nil {
		return fmt.Errorf("embed: %w", err)
	}
	if err := appendBlob("transformer.embeddings.packed", embBlob); err != nil {
		return err
	}
	dims.VocabSize = embBlob.Rows

	// Final / layer norms: MLX checkpoints already store the multiplicative γ used by
	// nn.RMSNorm (HF OffsetRMS (1+w) is applied at MLX sanitize time). Do not bake again.
	fn, err := hf.LoadF16Vector(stPath, index, "language_model.model.norm.weight")
	if err != nil {
		return fmt.Errorf("final_norm: %w", err)
	}
	if err := appendF32("transformer.final_norm", fn); err != nil {
		return err
	}

	// LM head
	lmBlob, err := hf.LoadMLX1BitMatrix(stPath, index, "language_model.lm_head")
	if err != nil {
		return fmt.Errorf("lm_head: %w", err)
	}
	if err := appendBlob("transformer.lm_head.packed", lmBlob); err != nil {
		return err
	}

	partialRotary := 1.0
	if rp, ok := cfg["rope_parameters"].(map[string]any); ok {
		if v, ok := rp["partial_rotary_factor"].(float64); ok {
			partialRotary = v
		}
	} else if v, ok := cfg["partial_rotary_factor"].(float64); ok {
		partialRotary = v
	}
	attnGate := false
	if v, ok := cfg["attn_output_gate"].(bool); ok {
		attnGate = v
	}

	for i := 0; i < dims.NumLayers; i++ {
		report(i+1, dims.NumLayers, fmt.Sprintf("layer %d (%s)", i, layerTypes[i]))
		prefix := fmt.Sprintf("blocks.%d", i)
		lp := fmt.Sprintf("language_model.model.layers.%d", i)

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

		// MLP (all layers)
		for _, role := range []struct{ hf, ent string }{
			{"mlp.gate_proj", "gate"},
			{"mlp.up_proj", "up"},
			{"mlp.down_proj", "down"},
		} {
			b, err := hf.LoadMLX1BitMatrix(stPath, index, lp+"."+role.hf)
			if err != nil {
				return fmt.Errorf("layer %d %s: %w", i, role.hf, err)
			}
			if err := appendBlob(prefix+"."+role.ent, b); err != nil {
				return err
			}
		}

		switch layerTypes[i] {
		case "full_attention":
			for _, role := range []struct{ hf, ent string }{
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
				return err
			}
			if err := appendF32(prefix+".q_norm", qn); err != nil {
				return err
			}
			kn, err := hf.LoadF16Vector(stPath, index, lp+".self_attn.k_norm.weight")
			if err != nil {
				return err
			}
			if err := appendF32(prefix+".k_norm", kn); err != nil {
				return err
			}
		case "linear_attention":
			for _, role := range []struct{ hf, ent string }{
				{"linear_attn.in_proj_qkv", "gdn_qkv"},
				{"linear_attn.in_proj_z", "gdn_z"},
				{"linear_attn.in_proj_b", "gdn_b"},
				{"linear_attn.in_proj_a", "gdn_a"},
				{"linear_attn.out_proj", "gdn_out"},
			} {
				b, err := hf.LoadMLX1BitMatrix(stPath, index, lp+"."+role.hf)
				if err != nil {
					return fmt.Errorf("layer %d %s: %w", i, role.hf, err)
				}
				if err := appendBlob(prefix+"."+role.ent, b); err != nil {
					return err
				}
			}
			// conv1d weight F16 [conv_dim, k, 1] — store flattened F32
			cw, err := hf.LoadF16Vector(stPath, index, lp+".linear_attn.conv1d.weight")
			if err != nil {
				return err
			}
			if err := appendF32(prefix+".gdn_conv", cw); err != nil {
				return err
			}
			aLog, err := hf.LoadF16Vector(stPath, index, lp+".linear_attn.A_log")
			if err != nil {
				return err
			}
			if err := appendF32(prefix+".gdn_A_log", aLog); err != nil {
				return err
			}
			dt, err := hf.LoadF16Vector(stPath, index, lp+".linear_attn.dt_bias")
			if err != nil {
				return err
			}
			if err := appendF32(prefix+".gdn_dt_bias", dt); err != nil {
				return err
			}
			gn, err := hf.LoadF16Vector(stPath, index, lp+".linear_attn.norm.weight")
			if err != nil {
				return err
			}
			if err := appendF32(prefix+".gdn_norm", gn); err != nil {
				return err
			}
		default:
			return fmt.Errorf("layer %d: unknown type %q", i, layerTypes[i])
		}
	}

	if err := payloadTmp.Close(); err != nil {
		return err
	}

	tokPath := filepath.Join(snapshotDir, "tokenizer.json")
	if _, err := os.Stat(tokPath); err != nil {
		tokPath = ""
	}

	rope := dims.RoPEFreqBase
	if rp, ok := cfg["rope_parameters"].(map[string]any); ok {
		if v, ok := rp["rope_theta"].(float64); ok && v > 0 {
			rope = v
		}
	}

	// Query dim includes output gate doubling when attn_output_gate.
	qDim := dims.NumHeads * dims.HeadDim
	if attnGate {
		qDim *= 2
	}

	spec := &TransformerSpec{
		Architecture: hf.ArchQwen35Hybrid.String(),
		HiddenSize:   dims.HiddenSize,
		VocabSize:    dims.VocabSize,
		LMHeadTied:   false,
		HasFinalNorm: true,
		WeightDType:  "PACKED",
		PackFormat:   quant.FormatBinaryPacked.String(),
		LMHeadPacked: true,
		Snapshot:     snapshotDir,
		Tokenizer:    tokPath,
		Repo:         opts.Repo,
		Engine:       "welvet",
		Dims: &TransformerDims{
			NumLayers:           dims.NumLayers,
			NumHeads:            dims.NumHeads,
			NumKVHeads:          dims.NumKVHeads,
			HeadDim:             dims.HeadDim,
			QueryDim:            qDim,
			KVDim:               dims.NumKVHeads * dims.HeadDim,
			IntermediateSize:    dims.IntermediateSize,
			RMSNormEps:          dims.RMSNormEps,
			RoPEFreqBase:        rope,
			PartialRotaryFactor: partialRotary,
			AttnOutputGate:      attnGate,
			LayerTypes:          layerTypes,
			LinearConvKernel:    lin.ConvKernel,
			LinearNumKeyHeads:   lin.NumKeyHeads,
			LinearNumValueHeads: lin.NumValueHeads,
			LinearKeyHeadDim:    lin.KeyHeadDim,
			LinearValueHeadDim:  lin.ValueHeadDim,
		},
	}

	doc := headerDoc{
		FormatVersion: FormatVersion,
		Engine:        "welvet",
		Status:        "packed",
		Transformer:   spec,
		Blobs:         blobs,
	}
	headerJSON, err := json.Marshal(doc)
	if err != nil {
		return err
	}
	if len(headerJSON) > headerMaxSize {
		return fmt.Errorf("entity header too large: %d", len(headerJSON))
	}

	out, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer out.Close()
	if _, err := out.Write([]byte(Magic)); err != nil {
		return err
	}
	var ver [2]byte
	binary.LittleEndian.PutUint16(ver[:], FormatVersion)
	if _, err := out.Write(ver[:]); err != nil {
		return err
	}
	if _, err := out.Write([]byte{0, 0}); err != nil {
		return err
	}
	var hlen [8]byte
	binary.LittleEndian.PutUint64(hlen[:], uint64(len(headerJSON)))
	if _, err := out.Write(hlen[:]); err != nil {
		return err
	}
	if _, err := out.Write(headerJSON); err != nil {
		return err
	}
	payload, err := os.Open(payloadPath)
	if err != nil {
		return err
	}
	defer payload.Close()
	if _, err := io.Copy(out, payload); err != nil {
		return err
	}
	return nil
}
