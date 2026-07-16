package quant

// Q8_0 block: 32 weights → f32 scale + 32×int8 (36 bytes).
const q8_0BlockWeights = 32
const q8_0BlockBytes = 36

// PackQ8_0 packs row-major float32 weights into a Q8_0 Blob.
func PackQ8_0(weights []float32, rows, cols int) (*Blob, error) {
	if err := checkShape("PackQ8_0", weights, rows, cols); err != nil {
		return nil, err
	}
	n := rows * cols
	blockCount := (n + q8_0BlockWeights - 1) / q8_0BlockWeights
	raw := make([]byte, blockCount*q8_0BlockBytes)
	for bi := 0; bi < blockCount; bi++ {
		start := bi * q8_0BlockWeights
		end := start + q8_0BlockWeights
		if end > n {
			end = n
		}
		maxAbs := maxAbsRange(weights, start, end)
		scale := maxAbs / 127
		if scale == 0 {
			scale = 1
		}
		off := bi * q8_0BlockBytes
		putF32(raw[off:], scale)
		for j := 0; j < q8_0BlockWeights; j++ {
			i := start + j
			var q int
			if i < n {
				q = clampInt(roundToInt(float64(weights[i])/float64(scale)), -127, 127)
			}
			raw[off+4+j] = byte(int8(q))
		}
	}
	return &Blob{
		Format:       FormatQ8_0,
		Rows:         rows,
		Cols:         cols,
		Raw:          raw,
		BlockWeights: q8_0BlockWeights,
	}, nil
}

func unpackQ8_0(b *Blob) ([]float32, error) {
	if b == nil || b.Format != FormatQ8_0 {
		return nil, errFormat("UnpackQ8_0", b)
	}
	n := b.Rows * b.Cols
	out := make([]float32, n)
	blockCount := len(b.Raw) / q8_0BlockBytes
	for bi := 0; bi < blockCount; bi++ {
		off := bi * q8_0BlockBytes
		scale := getF32(b.Raw[off:])
		start := bi * q8_0BlockWeights
		for j := 0; j < q8_0BlockWeights; j++ {
			i := start + j
			if i >= n {
				break
			}
			out[i] = float32(int8(b.Raw[off+4+j])) * scale
		}
	}
	return out, nil
}

func matVecQ8_0(b *Blob, x, y []float32) error {
	n := b.Rows * b.Cols
	for i := range y[:b.Rows] {
		y[i] = 0
	}
	blockCount := len(b.Raw) / q8_0BlockBytes
	for bi := 0; bi < blockCount; bi++ {
		off := bi * q8_0BlockBytes
		scale := getF32(b.Raw[off:])
		start := bi * q8_0BlockWeights
		for j := 0; j < q8_0BlockWeights; j++ {
			i := start + j
			if i >= n {
				break
			}
			y[i/b.Cols] += float32(int8(b.Raw[off+4+j])) * scale * x[i%b.Cols]
		}
	}
	return nil
}

func matVecTQ8_0(b *Blob, gy, gx []float32) error {
	n := b.Rows * b.Cols
	blockCount := len(b.Raw) / q8_0BlockBytes
	for bi := 0; bi < blockCount; bi++ {
		off := bi * q8_0BlockBytes
		scale := getF32(b.Raw[off:])
		start := bi * q8_0BlockWeights
		for j := 0; j < q8_0BlockWeights; j++ {
			i := start + j
			if i >= n {
				break
			}
			gx[i%b.Cols] += float32(int8(b.Raw[off+4+j])) * scale * gy[i/b.Cols]
		}
	}
	return nil
}
