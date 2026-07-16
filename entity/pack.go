package entity

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"

	"github.com/openfluke/welvet/hf"
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

// PackFromHF converts a Hugging Face snapshot directory to a Welvet .entity (FormatNone only).
func PackFromHF(snapshotDir, outPath string, opts PackOptions) error {
	if opts.Format != quant.FormatNone {
		return fmt.Errorf("entity.PackFromHF: only FormatNone supported this pass (got %s)", opts.Format.String())
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
		if err := appendF32("transformer.lm_head", globals.LMHead); err != nil {
			_ = payloadTmp.Close()
			return err
		}
	}

	layerFiles := hf.BuildLayerShardIndex(sts, dims.NumLayers)
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
		}{
			{prefix + ".attn_norm", block.AttnNorm},
			{prefix + ".q", block.Q},
			{prefix + ".k", block.K},
			{prefix + ".v", block.V},
			{prefix + ".o", block.O},
			{prefix + ".ffn_norm", block.FFNNorm},
			{prefix + ".gate", block.Gate},
			{prefix + ".up", block.Up},
			{prefix + ".down", block.Down},
		}
		for _, p := range pairs {
			if err := appendF32(p.name, p.data); err != nil {
				_ = payloadTmp.Close()
				return err
			}
		}
	}

	if err := payloadTmp.Close(); err != nil {
		return err
	}

	tokPath := filepath.Join(snapshotDir, "tokenizer.json")
	if _, err := os.Stat(tokPath); err != nil {
		tokPath = ""
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
	if _, err := out.Write([]byte{0, 0}); err != nil { // flags
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

type payloadAcc struct {
	w      *os.File
	offset uint64
}

func (a *payloadAcc) writeF32(data []float32) error {
	buf := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(buf[i*4:i*4+4], math.Float32bits(v))
	}
	n, err := a.w.Write(buf)
	if err != nil {
		return err
	}
	a.offset += uint64(n)
	return nil
}
