package entity

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/openfluke/welvet/model/hf"
	"github.com/openfluke/welvet/quant"
)

// VocabBlobPath is the embedded vocab.json path inside a wav2vec2 ENTITY.
const VocabBlobPath = "wav2vec2.vocab.json"

// ConfigBlobPath is the embedded config.json path inside a wav2vec2 ENTITY.
const ConfigBlobPath = "wav2vec2.config.json"

// PackFromWav2Vec2 converts a HF wav2vec2 CTC snapshot to a Welvet .entity (FormatNone FP32).
func PackFromWav2Vec2(snapshotDir, outPath string, opts PackOptions) error {
	if opts.Format != quant.FormatNone {
		return fmt.Errorf("entity.PackFromWav2Vec2: only FormatNone supported (got %s)", opts.Format.String())
	}
	configPath := filepath.Join(snapshotDir, "config.json")
	config, err := hf.LoadConfigJSON(configPath)
	if err != nil {
		return fmt.Errorf("config.json: %w", err)
	}
	if !hf.IsWav2Vec2CTC(config) {
		return fmt.Errorf("entity.PackFromWav2Vec2: not a wav2vec2 CTC config")
	}
	sts, err := filepath.Glob(filepath.Join(snapshotDir, "*.safetensors"))
	if err != nil {
		return err
	}
	if len(sts) == 0 {
		return fmt.Errorf("no *.safetensors in %s", snapshotDir)
	}
	if opts.Progress != nil {
		opts.Progress(0, 1, "loading wav2vec2 safetensors…")
	}
	tensors := make(map[string]hf.TensorWithMeta)
	for _, f := range sts {
		part, err := hf.LoadSafetensorsWithMeta(f, nil)
		if err != nil {
			return fmt.Errorf("load %s: %w", filepath.Base(f), err)
		}
		for k, v := range part {
			tensors[k] = v
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
	defer func() {
		payloadTmp.Close()
		os.Remove(payloadPath)
	}()
	acc := &payloadAcc{w: payloadTmp}
	var blobs []WeightBlob

	names := make([]string, 0, len(tensors))
	for k := range tensors {
		names = append(names, k)
	}
	sort.Strings(names)
	for i, name := range names {
		t := tensors[name]
		off := acc.offset
		if err := acc.writeF32(t.Data); err != nil {
			return err
		}
		blobs = append(blobs, WeightBlob{
			Path:   name,
			Offset: off,
			Length: uint64(len(t.Data) * 4),
			DType:  "FLOAT32",
			Format: "none",
			Shape:  append([]int(nil), t.Shape...),
			Native: true,
		})
		if opts.Progress != nil && (i == 0 || i+1 == len(names) || (i+1)%40 == 0) {
			opts.Progress(i+1, len(names), name)
		}
	}

	vocabBlob := ""
	if data, err := os.ReadFile(filepath.Join(snapshotDir, "vocab.json")); err == nil && len(data) > 0 {
		off := acc.offset
		if err := acc.writeBytes(data); err != nil {
			return err
		}
		blobs = append(blobs, WeightBlob{
			Path: VocabBlobPath, Offset: off, Length: uint64(len(data)),
			DType: "JSON", Format: "none", Native: true,
		})
		vocabBlob = VocabBlobPath
	}
	configBlob := ""
	if data, err := os.ReadFile(configPath); err == nil && len(data) > 0 {
		off := acc.offset
		if err := acc.writeBytes(data); err != nil {
			return err
		}
		blobs = append(blobs, WeightBlob{
			Path: ConfigBlobPath, Offset: off, Length: uint64(len(data)),
			DType: "JSON", Format: "none", Native: true,
		})
		configBlob = ConfigBlobPath
	}

	if err := payloadTmp.Close(); err != nil {
		return err
	}
	payload, err := os.ReadFile(payloadPath)
	if err != nil {
		return err
	}

	cfg := hf.EffectiveConfig(config)
	spec := &Wav2Vec2Spec{
		Architecture: "wav2vec2_ctc",
		HiddenSize:   hf.ConfigIntDefault(cfg, "hidden_size", 768),
		VocabSize:    hf.ConfigIntDefault(cfg, "vocab_size", 32),
		NumLayers:    hf.ConfigIntDefault(cfg, "num_hidden_layers", 12),
		NumHeads:     hf.ConfigIntDefault(cfg, "num_attention_heads", 12),
		Intermediate: hf.ConfigIntDefault(cfg, "intermediate_size", 3072),
		PadTokenID:   hf.ConfigIntDefault(cfg, "pad_token_id", 0),
		WeightDType:  "FLOAT32",
		PackFormat:   "none",
		Snapshot:     snapshotDir,
		Repo:         opts.Repo,
		VocabBlob:    vocabBlob,
		ConfigBlob:   configBlob,
	}
	return WriteWav2Vec2File(outPath, spec, blobs, payload)
}
