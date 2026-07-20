package entity

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/openfluke/welvet/model/hf"
	"github.com/openfluke/welvet/quant"
)

// ProgressFunc reports convert progress (blockIndex is 1-based; 0 = globals).
type ProgressFunc func(blockIndex, blockTotal int, detail string)

// PackOptions controls HF → ENTITY conversion.
type PackOptions struct {
	Repo     string
	Format   quant.Format
	Progress ProgressFunc
}

// PackFromHF converts a Hugging Face snapshot directory to a Welvet .entity.
// When opts.Format != FormatNone, decoder dense weights + LM head are baked as packed quant
// (embed + RMSNorm γ stay Float32 — Lucy parity).
func PackFromHF(snapshotDir, outPath string, opts PackOptions) error {
	if opts.Format != quant.FormatNone && !quant.Supported(opts.Format) {
		return fmt.Errorf("entity.PackFromHF: unsupported pack format %s", opts.Format.String())
	}

	configPath := filepath.Join(snapshotDir, "config.json")
	config, err := hf.LoadConfigJSON(configPath)
	if err != nil {
		return fmt.Errorf("config.json: %w", err)
	}
	kind := hf.DetectArchitecture(config)
	if kind == hf.ArchUnknown {
		return fmt.Errorf("unsupported HF architecture (model_type=%q)", hf.ConfigStringDefault(config, "model_type", ""))
	}
	if kind == hf.ArchQwen35Hybrid {
		return PackFromQwen35MLX(snapshotDir, outPath, opts)
	}
	// Dense Qwen3 / Bonsai-8B MLX 1-bit (no GDN).
	if bits, group, ok := hf.QuantBitsGroup(config); ok && bits == 1 && group == 128 {
		return PackFromQwen3MLX(snapshotDir, outPath, opts)
	}

	sts, err := filepath.Glob(filepath.Join(snapshotDir, "*.safetensors"))
	if err != nil {
		return err
	}
	if len(sts) == 0 {
		return fmt.Errorf("no *.safetensors in %s", snapshotDir)
	}

	dims, err := hf.ParseDecoderDims(config, sts)
	if err != nil {
		return err
	}
	if dims.VocabSize <= 0 {
		return fmt.Errorf("config missing vocab_size")
	}

	progress := opts.Progress
	if progress != nil {
		progress(0, dims.NumLayers, "loading globals…")
	}

	globalTensors := make(map[string][]float32)
	for _, f := range sts {
		part, err := hf.LoadSafetensorsSelective(f, hf.WeightIsGlobal)
		if err != nil {
			return fmt.Errorf("load globals from %s: %w", filepath.Base(f), err)
		}
		for k, v := range part {
			globalTensors[k] = v
		}
	}
	globals := hf.CloneGlobals(hf.MapGlobals(globalTensors))
	globalTensors = nil
	if len(globals.Embeddings) == 0 {
		return fmt.Errorf("no embeddings found in snapshot")
	}
	if dims.VocabSize*dims.HiddenSize != len(globals.Embeddings) {
		// allow mismatch only if config vocab wrong — prefer tensor size
		inferred := len(globals.Embeddings) / dims.HiddenSize
		if inferred > 0 && inferred*dims.HiddenSize == len(globals.Embeddings) {
			dims.VocabSize = inferred
		} else {
			return fmt.Errorf("embeddings len %d != vocab(%d)*hidden(%d)", len(globals.Embeddings), dims.VocabSize, dims.HiddenSize)
		}
	}

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
			Path:   path,
			Offset: off,
			Length: uint64(len(data) * 4),
			DType:  "FLOAT32",
			Format: "none",
			Native: false,
		})
		return nil
	}
	appendPacked := func(path string, data []float32, rows, cols int) error {
		if opts.Format == quant.FormatNone {
			return appendF32(path, data)
		}
		blob, err := quant.Pack(opts.Format, data, rows, cols)
		if err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		wire := EncodePackedBlob(blob)
		off := acc.offset
		if _, err := acc.w.Write(wire); err != nil {
			return err
		}
		acc.offset += uint64(len(wire))
		blobs = append(blobs, WeightBlob{
			Path:   path,
			Offset: off,
			Length: uint64(len(wire)),
			DType:  "PACKED",
			Format: opts.Format.String(),
			Rows:   rows,
			Cols:   cols,
			Native: true,
		})
		return nil
	}

	if err := appendF32("transformer.embeddings", globals.Embeddings); err != nil {
		_ = payloadTmp.Close()
		return err
	}
	if globals.HasFinalNorm {
		if err := appendF32("transformer.final_norm", globals.FinalNorm); err != nil {
			_ = payloadTmp.Close()
			return err
		}
	}
	if !globals.LMHeadTied {
		if err := appendPacked("transformer.lm_head", globals.LMHead, dims.VocabSize, dims.HiddenSize); err != nil {
			_ = payloadTmp.Close()
			return err
		}
	} else if opts.Format != quant.FormatNone {
		// Baked packed LM head for tied embeddings (+H parity).
		if err := appendPacked("transformer.lm_head.packed", globals.Embeddings, dims.VocabSize, dims.HiddenSize); err != nil {
			_ = payloadTmp.Close()
			return err
		}
	}

	layerFiles := hf.BuildLayerShardIndex(sts, dims.NumLayers)
	qDim := dims.NumHeads * dims.HeadDim
	kvDim := dims.NumKVHeads * dims.HeadDim
	if dims.QueryDim > 0 {
		qDim = dims.QueryDim
	}
	if dims.KVDim > 0 {
		kvDim = dims.KVDim
	}
	hidden := dims.HiddenSize
	inter := dims.IntermediateSize

	for i := 0; i < dims.NumLayers; i++ {
		if progress != nil {
			progress(i+1, dims.NumLayers, fmt.Sprintf("block %d/%d", i+1, dims.NumLayers))
		}
		tensors := make(map[string][]float32)
		for _, f := range layerFiles[i] {
			part, err := hf.LoadSafetensorsSelective(f, func(name string) bool {
				return hf.WeightMatchesLayer(name, i)
			})
			if err != nil {
				_ = payloadTmp.Close()
				return fmt.Errorf("layer %d from %s: %w", i, filepath.Base(f), err)
			}
			for k, v := range part {
				tensors[k] = v
			}
		}
		block, err := hf.MapBlock(tensors, i)
		if err != nil {
			_ = payloadTmp.Close()
			return err
		}
		prefix := fmt.Sprintf("blocks.%d", i)
		pairs := []struct {
			name string
			data []float32
			rows int
			cols int
		}{
			{prefix + ".attn_norm", block.AttnNorm, 0, 0},
			{prefix + ".q", block.Q, qDim, hidden},
			{prefix + ".k", block.K, kvDim, hidden},
			{prefix + ".v", block.V, kvDim, hidden},
			{prefix + ".o", block.O, hidden, qDim},
			{prefix + ".ffn_norm", block.FFNNorm, 0, 0},
			{prefix + ".gate", block.Gate, inter, hidden},
			{prefix + ".up", block.Up, inter, hidden},
			{prefix + ".down", block.Down, hidden, inter},
		}
		for _, p := range pairs {
			if strings.HasSuffix(p.name, "_norm") {
				if err := appendF32(p.name, p.data); err != nil {
					_ = payloadTmp.Close()
					return err
				}
				continue
			}
			if err := appendPacked(p.name, p.data, p.rows, p.cols); err != nil {
				_ = payloadTmp.Close()
				return err
			}
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

	spec := &TransformerSpec{
		Architecture: kind.String(),
		HiddenSize:   dims.HiddenSize,
		VocabSize:    dims.VocabSize,
		LMHeadTied:   globals.LMHeadTied,
		HasFinalNorm: globals.HasFinalNorm,
		WeightDType:  "FLOAT32",
		Snapshot:     snapshotDir,
		Tokenizer:    tokPath,
		Repo:         opts.Repo,
		Engine:       "welvet",
		Dims: &TransformerDims{
			NumLayers:        dims.NumLayers,
			NumHeads:         dims.NumHeads,
			NumKVHeads:       dims.NumKVHeads,
			HeadDim:          dims.HeadDim,
			QueryDim:         dims.QueryDim,
			KVDim:            dims.KVDim,
			IntermediateSize: dims.IntermediateSize,
			RMSNormEps:       dims.RMSNormEps,
			RoPEFreqBase:     dims.RoPEFreqBase,
		},
	}
	if opts.Format != quant.FormatNone {
		spec.PackFormat = opts.Format.String()
		spec.LMHeadPacked = true
	}

	payload, err := os.ReadFile(payloadPath)
	if err != nil {
		return err
	}
	return WriteTransformerFile(outPath, spec, blobs, payload)
}

type payloadAcc struct {
	w      *os.File
	offset uint64
}

func (a *payloadAcc) writeF32(data []float32) error {
	buf := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	n, err := a.w.Write(buf)
	a.offset += uint64(n)
	return err
}

func (a *payloadAcc) writeBytes(data []byte) error {
	n, err := a.w.Write(data)
	a.offset += uint64(n)
	return err
}

// appendTokenizerBlob embeds tokenizer.json into the payload when present.
// Returns the on-disk path recorded in the header (may be empty).
func appendTokenizerBlob(acc *payloadAcc, blobs *[]WeightBlob, snapshotDir string) string {
	tokPath := filepath.Join(snapshotDir, "tokenizer.json")
	data, err := os.ReadFile(tokPath)
	if err != nil || len(data) == 0 {
		return ""
	}
	off := acc.offset
	if err := acc.writeBytes(data); err != nil {
		return tokPath
	}
	*blobs = append(*blobs, WeightBlob{
		Path:   TokenizerBlobPath,
		Offset: off,
		Length: uint64(len(data)),
		DType:  "JSON",
		Format: "none",
		Native: true,
	})
	return tokPath
}
