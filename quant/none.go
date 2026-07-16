package quant

func packNone(weights []float32, rows, cols int) (*Blob, error) {
	if err := checkShape("PackNone", weights, rows, cols); err != nil {
		return nil, err
	}
	n := rows * cols
	raw := make([]byte, n*4)
	for i := 0; i < n; i++ {
		putF32(raw[i*4:], weights[i])
	}
	return &Blob{
		Format:       FormatNone,
		Rows:         rows,
		Cols:         cols,
		Raw:          raw,
		BlockWeights: 1,
	}, nil
}

func unpackNone(b *Blob) ([]float32, error) {
	if b == nil || b.Format != FormatNone {
		return nil, errFormat("UnpackNone", b)
	}
	n := b.Rows * b.Cols
	out := make([]float32, n)
	need := n * 4
	if len(b.Raw) < need {
		return nil, errShape("UnpackNone", b.Rows, b.Cols, len(b.Raw)/4)
	}
	for i := 0; i < n; i++ {
		out[i] = getF32(b.Raw[i*4:])
	}
	return out, nil
}

func matVecNone(b *Blob, x, y []float32) error {
	n := b.Rows * b.Cols
	if len(b.Raw) < n*4 {
		return errShape("MatVecNone", b.Rows, b.Cols, len(b.Raw)/4)
	}
	for r := 0; r < b.Rows; r++ {
		var sum float32
		row := r * b.Cols
		for c := 0; c < b.Cols; c++ {
			sum += getF32(b.Raw[(row+c)*4:]) * x[c]
		}
		y[r] = sum
	}
	return nil
}

func matVecTNone(b *Blob, gy, gx []float32) error {
	n := b.Rows * b.Cols
	if len(b.Raw) < n*4 {
		return errShape("MatVecTNone", b.Rows, b.Cols, len(b.Raw)/4)
	}
	for r := 0; r < b.Rows; r++ {
		row := r * b.Cols
		g := gy[r]
		for c := 0; c < b.Cols; c++ {
			gx[c] += getF32(b.Raw[(row+c)*4:]) * g
		}
	}
	return nil
}
