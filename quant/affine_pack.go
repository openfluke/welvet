package quant

import (
	"fmt"
	"math"
)

// packAffine encodes row-major float32 weights as FormatAffinePacked (4-bit, group 64).
func packAffine(weights []float32, rows, cols int) (*Blob, error) {
	return PackAffine4(weights, rows, cols, AffineG64Group)
}

// PackAffine4 packs float32 weights into AffinePacked (bits=4).
// Per group along cols: fit β + s·code with code∈[0,15] covering [min,max].
func PackAffine4(weights []float32, rows, cols, groupSize int) (*Blob, error) {
	if err := checkShape("PackAffine4", weights, rows, cols); err != nil {
		return nil, err
	}
	if groupSize <= 0 {
		groupSize = AffineG64Group
	}
	const bits = 4
	codesPerWord := 32 / bits
	if cols%codesPerWord != 0 {
		return nil, fmt.Errorf("PackAffine4: cols %d not multiple of %d", cols, codesPerWord)
	}
	if cols%groupSize != 0 {
		return nil, fmt.Errorf("PackAffine4: cols %d not multiple of group %d", cols, groupSize)
	}
	if groupSize%codesPerWord != 0 {
		return nil, fmt.Errorf("PackAffine4: group %d not multiple of codes/word %d", groupSize, codesPerWord)
	}

	wordsPerRow := cols / codesPerWord
	groupsPerRow := cols / groupSize
	wordsInGroup := groupSize / codesPerWord
	needSB := rows * groupsPerRow
	scales := make([]float32, needSB)
	biases := make([]float32, needSB)
	words := make([]uint32, rows*wordsPerRow)
	levels := float32((1 << bits) - 1) // 15

	for r := 0; r < rows; r++ {
		rowOff := r * cols
		for g := 0; g < groupsPerRow; g++ {
			colBase := g * groupSize
			mn := weights[rowOff+colBase]
			mx := mn
			for j := 1; j < groupSize; j++ {
				v := weights[rowOff+colBase+j]
				if v < mn {
					mn = v
				}
				if v > mx {
					mx = v
				}
			}
			s := (mx - mn) / levels
			if s == 0 || math.IsNaN(float64(s)) || math.IsInf(float64(s), 0) {
				s = 1
				mn = 0
			}
			si := r*groupsPerRow + g
			scales[si] = s
			biases[si] = mn
			invS := 1 / s
			for wi := 0; wi < wordsInGroup; wi++ {
				var word uint32
				base := colBase + wi*codesPerWord
				for j := 0; j < codesPerWord; j++ {
					v := weights[rowOff+base+j]
					code := int(math.Round(float64((v - mn) * invS)))
					if code < 0 {
						code = 0
					}
					if code > int(levels) {
						code = int(levels)
					}
					word |= uint32(code) << uint(j*bits)
				}
				words[r*wordsPerRow+g*wordsInGroup+wi] = word
			}
		}
	}
	return BlobFromMLXAffine4(words, scales, biases, rows, cols, groupSize, bits)
}
