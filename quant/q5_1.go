package quant

// Q5_1 block: 32 weights → f32 scale + f32 min + 20 bytes packed 5-bit (28 bytes).
// Reconstruction: w = min + q * scale, q ∈ [0,31].
const q5_1BlockWeights = 32
const q5_1BlockBytes = 28

func packQ5_1(weights []float32, rows, cols int) (*Blob, error) {
	if err := checkShape("PackQ5_1", weights, rows, cols); err != nil {
		return nil, err
	}
	n := rows * cols
	blockCount := (n + q5_1BlockWeights - 1) / q5_1BlockWeights
	raw := make([]byte, blockCount*q5_1BlockBytes)
	for bi := 0; bi < blockCount; bi++ {
		start := bi * q5_1BlockWeights
		end := start + q5_1BlockWeights
		if end > n {
			end = n
		}
		mn, mx := minMaxRange(weights, start, end)
		scale := (mx - mn) / 31
		if scale == 0 {
			scale = 1
		}
		off := bi * q5_1BlockBytes
		putF32(raw[off:], scale)
		putF32(raw[off+4:], mn)
		bw := &bitWriter{buf: raw[off+8 : off+q5_1BlockBytes]}
		for j := 0; j < q5_1BlockWeights; j++ {
			i := start + j
			q := 0
			if i < n {
				q = clampInt(roundToInt(float64((weights[i]-mn)/scale)), 0, 31)
			}
			bw.write(uint32(q), 5)
		}
	}
	return &Blob{
		Format:       FormatQ5_1,
		Rows:         rows,
		Cols:         cols,
		Raw:          raw,
		BlockWeights: q5_1BlockWeights,
	}, nil
}

func unpackQ5_1(b *Blob) ([]float32, error) {
	if b == nil || b.Format != FormatQ5_1 {
		return nil, errFormat("UnpackQ5_1", b)
	}
	n := b.Rows * b.Cols
	out := make([]float32, n)
	blockCount := len(b.Raw) / q5_1BlockBytes
	for bi := 0; bi < blockCount; bi++ {
		off := bi * q5_1BlockBytes
		scale := getF32(b.Raw[off:])
		mn := getF32(b.Raw[off+4:])
		br := &bitReader{buf: b.Raw[off+8 : off+q5_1BlockBytes]}
		start := bi * q5_1BlockWeights
		for j := 0; j < q5_1BlockWeights; j++ {
			i := start + j
			q := int(br.read(5))
			if i < n {
				out[i] = mn + float32(q)*scale
			}
		}
	}
	return out, nil
}

func matVecQ5_1(b *Blob, x, y []float32) error {
	n := b.Rows * b.Cols
	for i := range y[:b.Rows] {
		y[i] = 0
	}
	blockCount := len(b.Raw) / q5_1BlockBytes
	for bi := 0; bi < blockCount; bi++ {
		off := bi * q5_1BlockBytes
		scale := getF32(b.Raw[off:])
		mn := getF32(b.Raw[off+4:])
		br := &bitReader{buf: b.Raw[off+8 : off+q5_1BlockBytes]}
		start := bi * q5_1BlockWeights
		for j := 0; j < q5_1BlockWeights; j++ {
			i := start + j
			q := int(br.read(5))
			if i >= n {
				continue
			}
			y[i/b.Cols] += (mn + float32(q)*scale) * x[i%b.Cols]
		}
	}
	return nil
}

func matVecTQ5_1(b *Blob, gy, gx []float32) error {
	n := b.Rows * b.Cols
	blockCount := len(b.Raw) / q5_1BlockBytes
	for bi := 0; bi < blockCount; bi++ {
		off := bi * q5_1BlockBytes
		scale := getF32(b.Raw[off:])
		mn := getF32(b.Raw[off+4:])
		br := &bitReader{buf: b.Raw[off+8 : off+q5_1BlockBytes]}
		start := bi * q5_1BlockWeights
		for j := 0; j < q5_1BlockWeights; j++ {
			i := start + j
			q := int(br.read(5))
			if i >= n {
				continue
			}
			gx[i%b.Cols] += (mn + float32(q)*scale) * gy[i/b.Cols]
		}
	}
	return nil
}
