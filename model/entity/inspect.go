package entity

import (
	"fmt"
	"os"
)

// Info is a lightweight .entity probe (header only — no weight decode).
type Info struct {
	Path          string
	FormatVersion uint16
	Engine        string
	Status        string
	BlobCount     int
	HasTokenizer  bool
	Transformer   *TransformerSpec
}

// Inspect opens path, reads the header, and closes the file.
func Inspect(path string) (*Info, error) {
	ef, err := Open(path)
	if err != nil {
		return nil, err
	}
	defer ef.Close()
	hdr := ef.Header()
	if hdr == nil {
		return nil, fmt.Errorf("entity: empty header")
	}
	return &Info{
		Path:          path,
		FormatVersion: hdr.FormatVersion,
		Engine:        hdr.Engine,
		Status:        hdr.Status,
		BlobCount:     len(hdr.Blobs),
		HasTokenizer:  ef.HasTokenizerBlob(),
		Transformer:   hdr.Transformer,
	}, nil
}

// IsEntity reports whether path begins with the ENTITY magic.
func IsEntity(path string) bool {
	magic, err := PeekMagic(path)
	if err != nil {
		return false
	}
	return magic == Magic
}

// ImportFromHF is the public HF → Welvet .entity convert entrypoint
// (alias of PackFromHF). Prefer this name in app shells.
func ImportFromHF(snapshotDir, outPath string, opts PackOptions) error {
	if snapshotDir == "" || outPath == "" {
		return fmt.Errorf("entity.ImportFromHF: snapshotDir and outPath required")
	}
	if _, err := os.Stat(snapshotDir); err != nil {
		return fmt.Errorf("entity.ImportFromHF: snapshot: %w", err)
	}
	return PackFromHF(snapshotDir, outPath, opts)
}
