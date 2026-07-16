package quant

// FormatBinaryPacked:
//
//	32 weights / uint32, bit 0 → −scale, bit 1 → +scale
//	Per group scale = mean(|w|) in Scales[group]
//	Raw = little-endian uint32 words

const binaryGroup = 32

func packBinary(weights []float32, rows, cols int) (*Blob, error) {
	if err := checkShape("PackBinary", weights, rows, cols); err != nil {
		return nil, err
	}
	n := rows * cols
	nGroup := (n + binaryGroup - 1) / binaryGroup
	scales := make([]float32, nGroup)
	raw := make([]byte, nGroup*4)
	for g := 0; g < nGroup; g++ {
		start := g * binaryGroup
		end := start + binaryGroup
		if end > n {
			end = n
		}
		scale := absMean(weights, start, end)
		if scale == 0 {
			scale = 1
		}
		scales[g] = scale
		var word uint32
		for j := 0; j < binaryGroup; j++ {
			i := start + j
			if i < n && weights[i] >= 0 {
				word |= 1 << uint(j)
			}
		}
		putU32(raw[g*4:], word)
	}
	return &Blob{
		Format:       FormatBinaryPacked,
		Rows:         rows,
		Cols:         cols,
		Raw:          raw,
		Scales:       scales,
		BlockWeights: binaryGroup,
	}, nil
}

func forEachBinary(b *Blob, fn func(i int, w float32)) {
	if isBinaryG128(b) {
		forEachBinaryG128(b, fn)
		return
	}
	n := b.Rows * b.Cols
	nGroup := len(b.Raw) / 4
	for g := 0; g < nGroup; g++ {
		word := getU32(b.Raw[g*4:])
		scale := float32(1)
		if g < len(b.Scales) {
			scale = b.Scales[g]
		}
		start := g * binaryGroup
		for j := 0; j < binaryGroup; j++ {
			i := start + j
			if i >= n {
				break
			}
			if (word>>uint(j))&1 != 0 {
				fn(i, scale)
			} else {
				fn(i, -scale)
			}
		}
	}
}

func unpackBinary(b *Blob) ([]float32, error) {
	if b == nil || b.Format != FormatBinaryPacked {
		return nil, errFormat("UnpackBinary", b)
	}
	out := make([]float32, b.Rows*b.Cols)
	forEachBinary(b, func(i int, w float32) { out[i] = w })
	return out, nil
}

func matVecBinary(b *Blob, x, y []float32) error {
	if isBinaryG128(b) {
		return matVecBinaryG128(b, x, y)
	}
	for i := range y[:b.Rows] {
		y[i] = 0
	}
	forEachBinary(b, func(i int, w float32) {
		y[i/b.Cols] += w * x[i%b.Cols]
	})
	return nil
}

func matVecTBinary(b *Blob, gy, gx []float32) error {
	forEachBinary(b, func(i int, w float32) {
		gx[i%b.Cols] += w * gy[i/b.Cols]
	})
	return nil
}
