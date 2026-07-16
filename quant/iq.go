package quant

// Welvet IQ family (importance-matrix style), native grid + n-bit codes.
//
// Bits per weight (documented):
//
//	IQ1_S    — 1 bit   codes {0,1} → ±1 * scale
//	IQ2_XXS  — 2 bits  codes {0..3} → (q-1.5)*scale   (coarse: 1 scale / 32)
//	IQ2_XS   — 2 bits  same grid, finer scales (1 scale / 16)
//	IQ3_XXS  — 3 bits  codes {0..7} → (q-3.5)*scale   (1 scale / 32)
//	IQ3_S    — 3 bits  same, finer scales (1 scale / 16)
//	IQ4_NL   — 4 bits  non-linear lookup table * scale (1 scale / 32)
//	IQ4_XS   — 4 bits  linear (q-7.5)*scale           (1 scale / 16)
//
// Layout:
//
//	Scales[block] = f32 block scale
//	Raw           = tightly packed n-bit codes for all weights (padded to block)

const iqBlock = 32

// Approximate ggml IQ4_NL non-linear grid (normalized).
var iq4nlGrid = [16]float32{
	-1.0,
	-0.6961928,
	-0.52507305,
	-0.3949175,
	-0.28444138,
	-0.18477343,
	-0.091050036,
	0.0,
	0.0795803,
	0.1609302,
	0.2461123,
	0.33791524,
	0.44070983,
	0.562617,
	0.72295684,
	1.0,
}

type iqSpec struct {
	bits       int
	scaleGroup int // weights sharing one scale
	nonlinear  bool
	mid        float32 // subtract from q for linear grids; unused if nonlinear
}

func iqSpecFor(f Format) (iqSpec, bool) {
	switch f {
	case FormatIQ1_S:
		return iqSpec{bits: 1, scaleGroup: 32, mid: 0.5}, true
	case FormatIQ2_XXS:
		return iqSpec{bits: 2, scaleGroup: 32, mid: 1.5}, true
	case FormatIQ2_XS:
		return iqSpec{bits: 2, scaleGroup: 16, mid: 1.5}, true
	case FormatIQ3_XXS:
		return iqSpec{bits: 3, scaleGroup: 32, mid: 3.5}, true
	case FormatIQ3_S:
		return iqSpec{bits: 3, scaleGroup: 16, mid: 3.5}, true
	case FormatIQ4_NL:
		return iqSpec{bits: 4, scaleGroup: 32, nonlinear: true}, true
	case FormatIQ4_XS:
		return iqSpec{bits: 4, scaleGroup: 16, mid: 7.5}, true
	default:
		return iqSpec{}, false
	}
}

func packIQ(format Format, weights []float32, rows, cols int) (*Blob, error) {
	spec, ok := iqSpecFor(format)
	if !ok {
		return nil, ErrUnsupported(format, "Pack")
	}
	if err := checkShape("PackIQ", weights, rows, cols); err != nil {
		return nil, err
	}
	n := rows * cols
	// Pad to scaleGroup for clean packing.
	padN := ((n + spec.scaleGroup - 1) / spec.scaleGroup) * spec.scaleGroup
	nScale := padN / spec.scaleGroup
	scales := make([]float32, nScale)
	bw := newBitWriter(padN * spec.bits)
	qMax := (1 << spec.bits) - 1

	for s := 0; s < nScale; s++ {
		start := s * spec.scaleGroup
		end := start + spec.scaleGroup
		if end > n {
			end = n
		}
		if start >= n {
			scales[s] = 1
			for j := 0; j < spec.scaleGroup; j++ {
				bw.write(0, spec.bits)
			}
			continue
		}

		if spec.nonlinear {
			// Fit scale so maxAbs maps near grid edge (±1).
			amax := maxAbsRange(weights, start, end)
			scale := amax
			if scale == 0 {
				scale = 1
			}
			scales[s] = scale
			for j := 0; j < spec.scaleGroup; j++ {
				i := start + j
				best, bestErr := 0, float32(1e30)
				if i < n {
					target := weights[i] / scale
					for c := 0; c < 16; c++ {
						e := abs32(iq4nlGrid[c] - target)
						if e < bestErr {
							bestErr = e
							best = c
						}
					}
				}
				bw.write(uint32(best), 4)
			}
			continue
		}

		amax := maxAbsRange(weights, start, end)
		var scale float32
		if spec.bits == 1 {
			scale = amax
		} else {
			// Map so (qMax-mid)*scale ≈ amax
			denom := float32(qMax) - spec.mid
			if denom < 1 {
				denom = 1
			}
			scale = amax / denom
		}
		if scale == 0 {
			scale = 1
		}
		scales[s] = scale
		for j := 0; j < spec.scaleGroup; j++ {
			i := start + j
			q := 0
			if i < n {
				if spec.bits == 1 {
					if weights[i] >= 0 {
						q = 1
					}
				} else {
					q = clampInt(roundToInt(float64(weights[i])/float64(scale)+float64(spec.mid)), 0, qMax)
				}
			}
			bw.write(uint32(q), spec.bits)
		}
	}

	return &Blob{
		Format:       format,
		Rows:         rows,
		Cols:         cols,
		Raw:          bw.buf,
		Scales:       scales,
		BlockWeights: iqBlock,
		Meta:         []byte{byte(spec.bits), byte(spec.scaleGroup)},
	}, nil
}

func iqDequant(spec iqSpec, scale float32, q uint32) float32 {
	if spec.nonlinear {
		return scale * iq4nlGrid[q&15]
	}
	if spec.bits == 1 {
		if q&1 != 0 {
			return scale
		}
		return -scale
	}
	return scale * (float32(q) - spec.mid)
}

func forEachIQ(b *Blob, fn func(i int, w float32)) error {
	spec, ok := iqSpecFor(b.Format)
	if !ok {
		return ErrUnsupported(b.Format, "iq")
	}
	if len(b.Meta) >= 2 {
		spec.bits = int(b.Meta[0])
		spec.scaleGroup = int(b.Meta[1])
	}
	n := b.Rows * b.Cols
	br := &bitReader{buf: b.Raw}
	nScale := len(b.Scales)
	for s := 0; s < nScale; s++ {
		scale := b.Scales[s]
		start := s * spec.scaleGroup
		for j := 0; j < spec.scaleGroup; j++ {
			q := br.read(spec.bits)
			i := start + j
			if i >= n {
				continue
			}
			fn(i, iqDequant(spec, scale, q))
		}
	}
	return nil
}

func unpackIQ(b *Blob) ([]float32, error) {
	if b == nil {
		return nil, errFormat("UnpackIQ", b)
	}
	if _, ok := iqSpecFor(b.Format); !ok {
		return nil, errFormat("UnpackIQ", b)
	}
	out := make([]float32, b.Rows*b.Cols)
	err := forEachIQ(b, func(i int, w float32) { out[i] = w })
	return out, err
}

func matVecIQ(b *Blob, x, y []float32) error {
	for i := range y[:b.Rows] {
		y[i] = 0
	}
	return forEachIQ(b, func(i int, w float32) {
		y[i/b.Cols] += w * x[i%b.Cols]
	})
}

func matVecTIQ(b *Blob, gy, gx []float32) error {
	return forEachIQ(b, func(i int, w float32) {
		gx[i%b.Cols] += w * gy[i/b.Cols]
	})
}
