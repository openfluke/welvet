package quant

// Q4_1 block: 32 weights → f32 scale + f32 min + 16 nibble bytes (24 bytes).
// Reconstruction: w = min + q * scale, q ∈ [0,15].
const q4_1BlockWeights = 32
const q4_1BlockBytes = 24

func packQ4_1(weights []float32, rows, cols int) (*Blob, error) {
	if err := checkShape("PackQ4_1", weights, rows, cols); err != nil {
		return nil, err
	}
	n := rows * cols
	blockCount := (n + q4_1BlockWeights - 1) / q4_1BlockWeights
	raw := make([]byte, blockCount*q4_1BlockBytes)
	for bi := 0; bi < blockCount; bi++ {
		start := bi * q4_1BlockWeights
		end := start + q4_1BlockWeights
		if end > n {
			end = n
		}
		mn, mx := minMaxRange(weights, start, end)
		scale := (mx - mn) / 15
		if scale == 0 {
			scale = 1
		}
		off := bi * q4_1BlockBytes
		putF32(raw[off:], scale)
		putF32(raw[off+4:], mn)
		for j := 0; j < 16; j++ {
			var nib [2]byte
			for k := 0; k < 2; k++ {
				i := start + j*2 + k
				q := 0
				if i < n {
					q = clampInt(roundToInt(float64((weights[i]-mn)/scale)), 0, 15)
				}
				nib[k] = byte(q)
			}
			raw[off+8+j] = nib[0] | (nib[1] << 4)
		}
	}
	return &Blob{
		Format:       FormatQ4_1,
		Rows:         rows,
		Cols:         cols,
		Raw:          raw,
		BlockWeights: q4_1BlockWeights,
	}, nil
}

func unpackQ4_1(b *Blob) ([]float32, error) {
	if b == nil || b.Format != FormatQ4_1 {
		return nil, errFormat("UnpackQ4_1", b)
	}
	n := b.Rows * b.Cols
	out := make([]float32, n)
	blockCount := len(b.Raw) / q4_1BlockBytes
	for bi := 0; bi < blockCount; bi++ {
		off := bi * q4_1BlockBytes
		scale := getF32(b.Raw[off:])
		mn := getF32(b.Raw[off+4:])
		start := bi * q4_1BlockWeights
		for j := 0; j < 16; j++ {
			w := b.Raw[off+8+j]
			for k := 0; k < 2; k++ {
				i := start + j*2 + k
				if i >= n {
					break
				}
				q := int((w >> (4 * k)) & 0xF)
				out[i] = mn + float32(q)*scale
			}
		}
	}
	return out, nil
}

func matVecQ4_1(b *Blob, x, y []float32) error {
	n := b.Rows * b.Cols
	for i := range y[:b.Rows] {
		y[i] = 0
	}
	blockCount := len(b.Raw) / q4_1BlockBytes
	for bi := 0; bi < blockCount; bi++ {
		off := bi * q4_1BlockBytes
		scale := getF32(b.Raw[off:])
		mn := getF32(b.Raw[off+4:])
		start := bi * q4_1BlockWeights
		for j := 0; j < 16; j++ {
			w := b.Raw[off+8+j]
			for k := 0; k < 2; k++ {
				i := start + j*2 + k
				if i >= n {
					break
				}
				q := int((w >> (4 * k)) & 0xF)
				y[i/b.Cols] += (mn + float32(q)*scale) * x[i%b.Cols]
			}
		}
	}
	return nil
}

func matVecTQ4_1(b *Blob, gy, gx []float32) error {
	n := b.Rows * b.Cols
	blockCount := len(b.Raw) / q4_1BlockBytes
	for bi := 0; bi < blockCount; bi++ {
		off := bi * q4_1BlockBytes
		scale := getF32(b.Raw[off:])
		mn := getF32(b.Raw[off+4:])
		start := bi * q4_1BlockWeights
		for j := 0; j < 16; j++ {
			w := b.Raw[off+8+j]
			for k := 0; k < 2; k++ {
				i := start + j*2 + k
				if i >= n {
					break
				}
				q := int((w >> (4 * k)) & 0xF)
				gx[i%b.Cols] += (mn + float32(q)*scale) * gy[i/b.Cols]
			}
		}
	}
	return nil
}
