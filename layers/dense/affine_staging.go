package dense

import (
	"encoding/binary"

	"github.com/openfluke/welvet/quant"
)

// AffineBlobStaging extracts scales, biases (β), and u32 words for WebGPU Affine4 GEMV.
func AffineBlobStaging(b *quant.Blob) (scales, biases []float32, words []uint32, group int, ok bool) {
	if b == nil || !quant.IsAffinePacked(b) {
		return nil, nil, nil, 0, false
	}
	group = b.BlockWeights
	if group <= 0 {
		group = quant.AffineG64Group
	}
	if b.Cols%group != 0 || b.Cols%8 != 0 {
		return nil, nil, nil, 0, false
	}
	needWords := (b.Rows * b.Cols) / 8
	needSB := b.Rows * (b.Cols / group)
	if len(b.Scales) < needSB || len(b.Mins) < needSB || len(b.Raw) < needWords*4 {
		return nil, nil, nil, 0, false
	}
	words = make([]uint32, needWords)
	for i := 0; i < needWords; i++ {
		words[i] = binary.LittleEndian.Uint32(b.Raw[i*4:])
	}
	return b.Scales[:needSB], b.Mins[:needSB], words, group, true
}
