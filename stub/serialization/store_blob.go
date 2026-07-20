package serialization

import (
	"encoding/base64"
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/weights"
)

func encodeStore(name string, s *weights.Store) (StoreBlob, error) {
	if s == nil {
		return StoreBlob{}, fmt.Errorf("serialization: nil store %q", name)
	}
	snap, err := weights.TakeSnapshot(s)
	if err != nil {
		return StoreBlob{}, err
	}
	b := StoreBlob{
		Name:   name,
		DType:  snap.DType.String(),
		Format: snap.Format.String(),
		Rows:   snap.Rows,
		Cols:   snap.Cols,
		Data:   base64.StdEncoding.EncodeToString(snap.Raw),
		Scale:  snap.Scale,
		Native: true,
	}
	if len(snap.Bias) > 0 {
		b.Bias = base64.StdEncoding.EncodeToString(weights.EncodeF64LE(snap.Bias))
	}
	return b, nil
}

func decodeStore(b StoreBlob) (*weights.Store, error) {
	raw, err := base64.StdEncoding.DecodeString(b.Data)
	if err != nil {
		return nil, fmt.Errorf("serialization: store %q data: %w", b.Name, err)
	}
	var bias []float64
	if b.Bias != "" {
		br, err := base64.StdEncoding.DecodeString(b.Bias)
		if err != nil {
			return nil, fmt.Errorf("serialization: store %q bias: %w", b.Name, err)
		}
		bias, err = weights.DecodeF64LE(br)
		if err != nil {
			return nil, err
		}
	}
	format := quant.ParseFormatName(b.Format)
	dt := core.ParseDType(b.DType)
	return weights.Restore(weights.Snapshot{
		DType:  dt,
		Format: format,
		Rows:   b.Rows,
		Cols:   b.Cols,
		Scale:  b.Scale,
		Raw:    raw,
		Bias:   bias,
	})
}

func storeMap(blobs []StoreBlob) map[string]StoreBlob {
	m := make(map[string]StoreBlob, len(blobs))
	for _, b := range blobs {
		m[b.Name] = b
	}
	return m
}

func mustStore(m map[string]StoreBlob, name string) (*weights.Store, error) {
	b, ok := m[name]
	if !ok {
		return nil, fmt.Errorf("serialization: missing store %q", name)
	}
	return decodeStore(b)
}

func denseFromStore(in, out int, act core.ActivationType, s *weights.Store) (*dense.Layer, error) {
	if s == nil {
		return nil, fmt.Errorf("serialization: nil dense store")
	}
	if s.Cols != in || s.Rows != out {
		// Prefer store geometry when present.
		in, out = s.Cols, s.Rows
	}
	l, err := dense.New(in, out, act, s.DType)
	if err != nil {
		return nil, err
	}
	l.Weights = s
	l.Core.DType = s.DType
	return l, nil
}

func encodeF32Extra(name string, v []float32) (string, error) {
	if v == nil {
		return "", nil
	}
	return base64.StdEncoding.EncodeToString(weights.EncodeF32LE(v)), nil
}

func decodeF32Extra(extras map[string]string, name string) ([]float32, error) {
	if extras == nil {
		return nil, nil
	}
	s, ok := extras[name]
	if !ok || s == "" {
		return nil, nil
	}
	raw, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		return nil, err
	}
	if len(raw)%4 != 0 {
		return nil, fmt.Errorf("serialization: extra %q bad length", name)
	}
	return weights.DecodeF32LE(raw, len(raw)/4)
}

func encodeQuantBlob(name string, b *quant.Blob) (StoreBlob, error) {
	if b == nil {
		return StoreBlob{}, fmt.Errorf("serialization: nil quant blob %q", name)
	}
	return StoreBlob{
		Name:   name,
		DType:  core.DTypeFloat32.String(),
		Format: b.Format.String(),
		Rows:   b.Rows,
		Cols:   b.Cols,
		Data:   base64.StdEncoding.EncodeToString(weights.EncodeBlobWire(b)),
		Scale:  1,
		Native: true,
	}, nil
}

func decodeQuantBlob(b StoreBlob) (*quant.Blob, error) {
	raw, err := base64.StdEncoding.DecodeString(b.Data)
	if err != nil {
		return nil, err
	}
	return weights.DecodeBlobWire(quant.ParseFormatName(b.Format), b.Rows, b.Cols, raw)
}
