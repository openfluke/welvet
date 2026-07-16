package quant

// FormatTernaryPacked (BitNet 1.58 style):
//
//	16 weights / uint32, 2-bit codes {0,1,2} → {-1, 0, +1}
//	Per group scale = mean(|w|) stored in Scales[group]
//	Raw = little-endian uint32 words

const ternaryGroup = 16

func packTernary(weights []float32, rows, cols int) (*Blob, error) {
	if err := checkShape("PackTernary", weights, rows, cols); err != nil {
		return nil, err
	}
	n := rows * cols
	nGroup := (n + ternaryGroup - 1) / ternaryGroup
	scales := make([]float32, nGroup)
	raw := make([]byte, nGroup*4)
	for g := 0; g < nGroup; g++ {
		start := g * ternaryGroup
		end := start + ternaryGroup
		if end > n {
			end = n
		}
		scale := absMean(weights, start, end)
		if scale == 0 {
			scale = 1
		}
		scales[g] = scale
		var word uint32
		for j := 0; j < ternaryGroup; j++ {
			i := start + j
			code := uint32(1) // 0
			if i < n {
				q := roundToInt(float64(weights[i] / scale))
				q = clampInt(q, -1, 1)
				code = uint32(q + 1) // -1,0,+1 → 0,1,2
			}
			word |= (code & 3) << uint(j*2)
		}
		putU32(raw[g*4:], word)
	}
	return &Blob{
		Format:       FormatTernaryPacked,
		Rows:         rows,
		Cols:         cols,
		Raw:          raw,
		Scales:       scales,
		BlockWeights: ternaryGroup,
	}, nil
}

func ternaryCodeToVal(code uint32, scale float32) float32 {
	switch code & 3 {
	case 0:
		return -scale
	case 2:
		return scale
	default:
		return 0
	}
}

func forEachTernary(b *Blob, fn func(i int, w float32)) {
	n := b.Rows * b.Cols
	nGroup := len(b.Raw) / 4
	for g := 0; g < nGroup; g++ {
		word := getU32(b.Raw[g*4:])
		scale := float32(1)
		if g < len(b.Scales) {
			scale = b.Scales[g]
		}
		start := g * ternaryGroup
		for j := 0; j < ternaryGroup; j++ {
			i := start + j
			if i >= n {
				break
			}
			fn(i, ternaryCodeToVal(word>>uint(j*2), scale))
		}
	}
}

func unpackTernary(b *Blob) ([]float32, error) {
	if b == nil || b.Format != FormatTernaryPacked {
		return nil, errFormat("UnpackTernary", b)
	}
	out := make([]float32, b.Rows*b.Cols)
	forEachTernary(b, func(i int, w float32) { out[i] = w })
	return out, nil
}

func matVecTernary(b *Blob, x, y []float32) error {
	for i := range y[:b.Rows] {
		y[i] = 0
	}
	forEachTernary(b, func(i int, w float32) {
		y[i/b.Cols] += w * x[i%b.Cols]
	})
	return nil
}

func matVecTTernary(b *Blob, gy, gx []float32) error {
	forEachTernary(b, func(i int, w float32) {
		gx[i%b.Cols] += w * gy[i/b.Cols]
	})
	return nil
}
