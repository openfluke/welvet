package wav2vec2

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/openfluke/welvet/model/entity"
	"github.com/openfluke/welvet/model/hf"
)

// LoadEntity opens a Welvet wav2vec2 CTC .entity and builds a Model.
func LoadEntity(path string) (*Model, error) {
	ef, err := entity.Open(path)
	if err != nil {
		return nil, err
	}
	defer ef.Close()
	hdr := ef.Header()
	if hdr == nil || hdr.Wav2Vec2 == nil {
		return nil, fmt.Errorf("wav2vec2: %s is not a wav2vec2 ENTITY", path)
	}
	spec := hdr.Wav2Vec2

	var cfg Config
	if spec.ConfigBlob != "" {
		raw, err := ef.LoadBlobBytes(spec.ConfigBlob)
		if err != nil {
			return nil, err
		}
		if err := json.Unmarshal(raw, &cfg); err != nil {
			return nil, fmt.Errorf("wav2vec2: config blob: %w", err)
		}
	} else {
		cfg = Base960h()
		cfg.HiddenSize = spec.HiddenSize
		cfg.NumHiddenLayers = spec.NumLayers
		cfg.NumAttentionHeads = spec.NumHeads
		cfg.IntermediateSize = spec.Intermediate
		cfg.VocabSize = spec.VocabSize
		cfg.PadTokenID = spec.PadTokenID
	}
	if cfg.HiddenSize == 0 {
		cfg = Base960h()
	}

	vocabPath := spec.VocabBlob
	if vocabPath == "" {
		vocabPath = entity.VocabBlobPath
	}
	vocabRaw, err := ef.LoadBlobBytes(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("wav2vec2: vocab: %w", err)
	}
	vocab, err := LoadVocabBytes(vocabRaw, cfg.PadTokenID)
	if err != nil {
		return nil, err
	}

	tensors := make(map[string]hf.TensorWithMeta)
	for _, b := range hdr.Blobs {
		if b.DType == "JSON" || b.Path == entity.VocabBlobPath || b.Path == entity.ConfigBlobPath {
			continue
		}
		data, err := ef.LoadBlob(b.Path)
		if err != nil {
			return nil, fmt.Errorf("wav2vec2: blob %s: %w", b.Path, err)
		}
		shape := append([]int(nil), b.Shape...)
		if len(shape) == 0 && b.Rows > 0 && b.Cols > 0 {
			shape = []int{b.Rows, b.Cols}
		}
		tensors[b.Path] = hf.TensorWithMeta{Shape: shape, Data: data}
	}
	return BuildFromTensors(cfg, vocab, tensors)
}

// LoadAuto loads from a .entity path or an HF snapshot directory.
func LoadAuto(path string) (*Model, error) {
	st, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if !st.IsDir() {
		return LoadEntity(path)
	}
	// Prefer sibling .entity next to snapshot if present? caller decides.
	if _, err := os.Stat(filepath.Join(path, "model.safetensors")); err == nil {
		return LoadHFDir(path)
	}
	return nil, fmt.Errorf("wav2vec2: %s is neither .entity nor HF snapshot", path)
}
