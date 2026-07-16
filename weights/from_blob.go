package weights

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
)

// FromBlob builds a Store from a packed quant.Blob (ENTITY baked quant).
func FromBlob(b *quant.Blob) (*Store, error) {
	if b == nil {
		return nil, fmt.Errorf("weights: nil blob")
	}
	if b.Rows <= 0 || b.Cols <= 0 {
		return nil, fmt.Errorf("weights: bad blob shape %dx%d", b.Rows, b.Cols)
	}
	if b.Format == quant.FormatNone {
		return nil, fmt.Errorf("weights: blob format is none")
	}
	if b.Format == quant.FormatQ4_0 {
		quant.EnsureQ4SIMDCache(b)
	}
	return &Store{
		DType:  core.DTypeFloat32,
		Format: b.Format,
		Rows:   b.Rows,
		Cols:   b.Cols,
		Packed: b,
		Scale:  1,
	}, nil
}
