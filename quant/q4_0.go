package quant

import "math"

// Q4_0 block: 32 weights → 4-byte f32 scale + 16 bytes nibbles (20 bytes).
const q4_0BlockWeights = 32
const q4_0BlockBytes = 20

// PackQ4_0 packs row-major float32 weights into a Q4_0 Blob.
func PackQ4_0(weights []float32, rows, cols int) (*Blob, error) {
	if err := checkShape("PackQ4_0", weights, rows, cols); err != nil {
		return nil, err
	}
	n := rows * cols
	blockCount := (n + q4_0BlockWeights - 1) / q4_0BlockWeights
	raw := make([]byte, blockCount*q4_0BlockBytes)
	for bi := 0; bi < blockCount; bi++ {
		start := bi * q4_0BlockWeights
		end := start + q4_0BlockWeights
		if end > n {
			end = n
		}
		maxAbs := maxAbsRange(weights, start, end)
		scale := maxAbs / 7
		if scale == 0 {
			scale = 1
		}
		off := bi * q4_0BlockBytes
		putF32(raw[off:], scale)
		for j := 0; j < 16; j++ {
			i1 := start + j*2
			i2 := start + j*2 + 1
			q1 := quantNibble(weights, i1, n, scale)
			q2 := quantNibble(weights, i2, n, scale)
			raw[off+4+j] = byte(q1&0xF) | byte((q2&0xF)<<4)
		}
	}
	return &Blob{
		Format:       FormatQ4_0,
		Rows:         rows,
		Cols:         cols,
		Raw:          raw,
		BlockWeights: q4_0BlockWeights,
	}, nil
}

func UnpackQ4_0(b *Blob) ([]float32, error) {
	if b == nil || b.Format != FormatQ4_0 {
		return nil, errFormat("UnpackQ4_0", b)
	}
	n := b.Rows * b.Cols
	out := make([]float32, n)
	blockCount := len(b.Raw) / q4_0BlockBytes
	for bi := 0; bi < blockCount; bi++ {
		off := bi * q4_0BlockBytes
		scale := getF32(b.Raw[off:])
		start := bi * q4_0BlockWeights
		for j := 0; j < 16; j++ {
			w := b.Raw[off+4+j]
			q1 := int8(w & 0xF)
			if q1 > 7 {
				q1 -= 16
			}
			q2 := int8(w >> 4)
			if q2 > 7 {
				q2 -= 16
			}
			i1 := start + j*2
			i2 := start + j*2 + 1
			if i1 < n {
				out[i1] = float32(q1) * scale
			}
			if i2 < n {
				out[i2] = float32(q2) * scale
			}
		}
	}
	return out, nil
}

func matVecQ4_0(b *Blob, x, y []float32) error {
	n := b.Rows * b.Cols
	blockCount := len(b.Raw) / q4_0BlockBytes
	for i := range y[:b.Rows] {
		y[i] = 0
	}
	for bi := 0; bi < blockCount; bi++ {
		off := bi * q4_0BlockBytes
		scale := getF32(b.Raw[off:])
		start := bi * q4_0BlockWeights
		for j := 0; j < 16; j++ {
			w := b.Raw[off+4+j]
			q1 := int8(w & 0xF)
			if q1 > 7 {
				q1 -= 16
			}
			q2 := int8(w >> 4)
			if q2 > 7 {
				q2 -= 16
			}
			for k, q := range [2]int8{q1, q2} {
				i := start + j*2 + k
				if i >= n {
					break
				}
				y[i/b.Cols] += float32(q) * scale * x[i%b.Cols]
			}
		}
	}
	return nil
}

func matVecTQ4_0(b *Blob, gy, gx []float32) error {
	n := b.Rows * b.Cols
	blockCount := len(b.Raw) / q4_0BlockBytes
	for bi := 0; bi < blockCount; bi++ {
		off := bi * q4_0BlockBytes
		scale := getF32(b.Raw[off:])
		start := bi * q4_0BlockWeights
		for j := 0; j < 16; j++ {
			w := b.Raw[off+4+j]
			q1 := int8(w & 0xF)
			if q1 > 7 {
				q1 -= 16
			}
			q2 := int8(w >> 4)
			if q2 > 7 {
				q2 -= 16
			}
			for k, q := range [2]int8{q1, q2} {
				i := start + j*2 + k
				if i >= n {
					break
				}
				gx[i%b.Cols] += float32(q) * scale * gy[i/b.Cols]
			}
		}
	}
	return nil
}

func quantNibble(w []float32, i, n int, scale float32) int8 {
	if i >= n {
		return 0
	}
	q := int8(math.Round(float64(w[i] / scale)))
	if q > 7 {
		q = 7
	}
	if q < -8 {
		q = -8
	}
	return q
}
