package entity

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// WriteTransformerFile writes a packed transformer .entity checkpoint.
// payload is the concatenated blob section (offsets in blobs are relative to payload start).
func WriteTransformerFile(outPath string, spec *TransformerSpec, blobs []WeightBlob, payload []byte) error {
	if spec == nil {
		return fmt.Errorf("entity.WriteTransformerFile: nil spec")
	}
	doc := headerDoc{
		FormatVersion: FormatVersion,
		Engine:        "welvet",
		Status:        "packed",
		Transformer:   spec,
		Blobs:         blobs,
	}
	if spec.Engine != "" {
		doc.Engine = spec.Engine
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
	if len(payload) > 0 {
		if _, err := out.Write(payload); err != nil {
			return err
		}
	}
	return nil
}

// WriteTransformerFileFromReader is WriteTransformerFile with a streaming payload.
func WriteTransformerFileFromReader(outPath string, spec *TransformerSpec, blobs []WeightBlob, payload io.Reader) error {
	tmp, err := os.CreateTemp("", "welvet-entity-payload-*.bin")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if payload != nil {
		if _, err := io.Copy(tmp, payload); err != nil {
			tmp.Close()
			return err
		}
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	data, err := os.ReadFile(tmpPath)
	if err != nil {
		return err
	}
	return WriteTransformerFile(outPath, spec, blobs, data)
}
