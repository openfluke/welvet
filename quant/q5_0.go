package quant

// Q5_0 block: 32 weights → f32 scale + 20 bytes of packed 5-bit codes (24 bytes).
// Codes are signed via offset: q_stored ∈ [0,31], value = (q_stored - 16) * scale.
const q5_0BlockWeights = 32
const q5_0BlockBytes = 24

func packQ5_0(weights []float32, rows, cols int) (*Blob, error) {
	if err := checkShape("PackQ5_0", weights, rows, cols); err != nil {
		return nil, err
	}
	n := rows * cols
	blockCount := (n + q5_0BlockWeights - 1) / q5_0BlockWeights
	raw := make([]byte, blockCount*q5_0BlockBytes)
	for bi := 0; bi < blockCount; bi++ {
		start := bi * q5_0BlockWeights
		end := start + q5_0BlockWeights
		if end > n {
			end = n
		}
		maxAbs := maxAbsRange(weights, start, end)
		scale := maxAbs / 15
		if scale == 0 {
			scale = 1
		}
		off := bi * q5_0BlockBytes
		putF32(raw[off:], scale)
		bw := &bitWriter{buf: raw[off+4 : off+q5_0BlockBytes]}
		for j := 0; j < q5_0BlockWeights; j++ {
			i := start + j
			q := 16
			if i < n {
				q = clampInt(roundToInt(float64(weights[i])/float64(scale))+16, 0, 31)
			}
			bw.write(uint32(q), 5)
		}
	}
	return &Blob{
		Format:       FormatQ5_0,
		Rows:         rows,
		Cols:         cols,
		Raw:          raw,
		BlockWeights: q5_0BlockWeights,
	}, nil
}

func unpackQ5_0(b *Blob) ([]float32, error) {
	if b == nil || b.Format != FormatQ5_0 {
		return nil, errFormat("UnpackQ5_0", b)
	}
	n := b.Rows * b.Cols
	out := make([]float32, n)
	blockCount := len(b.Raw) / q5_0BlockBytes
	for bi := 0; bi < blockCount; bi++ {
		off := bi * q5_0BlockBytes
		scale := getF32(b.Raw[off:])
		br := &bitReader{buf: b.Raw[off+4 : off+q5_0BlockBytes]}
		start := bi * q5_0BlockWeights
		for j := 0; j < q5_0BlockWeights; j++ {
			i := start + j
			q := int(br.read(5)) - 16
			if i < n {
				out[i] = float32(q) * scale
			}
		}
	}
	return out, nil
}

func matVecQ5_0(b *Blob, x, y []float32) error {
	n := b.Rows * b.Cols
	for i := range y[:b.Rows] {
		y[i] = 0
	}
	blockCount := len(b.Raw) / q5_0BlockBytes
	for bi := 0; bi < blockCount; bi++ {
		off := bi * q5_0BlockBytes
		scale := getF32(b.Raw[off:])
		br := &bitReader{buf: b.Raw[off+4 : off+q5_0BlockBytes]}
		start := bi * q5_0BlockWeights
		for j := 0; j < q5_0BlockWeights; j++ {
			i := start + j
			q := int(br.read(5)) - 16
			if i >= n {
				continue
			}
			y[i/b.Cols] += float32(q) * scale * x[i%b.Cols]
		}
	}
	return nil
}

func matVecTQ5_0(b *Blob, gy, gx []float32) error {
	n := b.Rows * b.Cols
	blockCount := len(b.Raw) / q5_0BlockBytes
	for bi := 0; bi < blockCount; bi++ {
		off := bi * q5_0BlockBytes
		scale := getF32(b.Raw[off:])
		br := &bitReader{buf: b.Raw[off+4 : off+q5_0BlockBytes]}
		start := bi * q5_0BlockWeights
		for j := 0; j < q5_0BlockWeights; j++ {
			i := start + j
			q := int(br.read(5)) - 16
			if i >= n {
				continue
			}
			gx[i%b.Cols] += float32(q) * scale * gy[i/b.Cols]
		}
	}
	return nil
}
