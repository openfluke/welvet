package quant

// Welvet k-quant superblocks (ggml-inspired, not byte-identical to llama.cpp).
//
// QK = 256 weights per superblock, split into 16 groups of 16.
//
// Layout per superblock in Raw:
//
//	[0:4]   d     float32  — outer scale
//	[4:8]   dmin  float32  — outer min scale (0 when unused)
//	[8:24]  scales[16] uint8 — group scales relative to d (0..255 → d*s/255)
//	[24:40] mins[16] uint8   — only if hasDmin (asymmetric Q2_K..Q5_K)
//	[…]     qs               — tightly packed group quants, bitsPerWeight each
//
// Asymmetric (Q2_K, Q3_K, Q4_K, Q5_K):
//
//	w = dmin*(mins[g]/255) + d*(scales[g]/255) * q ,  q ∈ [0, qMax]
//
// Symmetric-ish Q6_K (no mins plane):
//
//	w = d*(scales[g]/255) * (q - 32) ,  q ∈ [0,63]

const (
	kQK     = 256
	kGroup  = 16
	kGroups = kQK / kGroup // 16
)

type kSpec struct {
	bits    int
	hasDmin bool
	qMax    int
	mid     int // subtract for symmetric; 0 if asymmetric
}

func kSpecFor(f Format) (kSpec, bool) {
	switch f {
	case FormatQ2_K:
		return kSpec{bits: 2, hasDmin: true, qMax: 3}, true
	case FormatQ3_K:
		return kSpec{bits: 3, hasDmin: true, qMax: 7}, true
	case FormatQ4_K:
		return kSpec{bits: 4, hasDmin: true, qMax: 15}, true
	case FormatQ5_K:
		return kSpec{bits: 5, hasDmin: true, qMax: 31}, true
	case FormatQ6_K:
		return kSpec{bits: 6, hasDmin: false, qMax: 63, mid: 32}, true
	default:
		return kSpec{}, false
	}
}

func kSuperBytes(spec kSpec) int {
	n := 8 + kGroups // d + dmin + scales
	if spec.hasDmin {
		n += kGroups
	}
	n += (kQK*spec.bits + 7) / 8
	return n
}

func packK(format Format, weights []float32, rows, cols int) (*Blob, error) {
	spec, ok := kSpecFor(format)
	if !ok {
		return nil, ErrUnsupported(format, "Pack")
	}
	if err := checkShape("PackK", weights, rows, cols); err != nil {
		return nil, err
	}
	n := rows * cols
	sbCount := (n + kQK - 1) / kQK
	sbBytes := kSuperBytes(spec)
	raw := make([]byte, sbCount*sbBytes)

	for si := 0; si < sbCount; si++ {
		base := si * kQK
		off := si * sbBytes

		var groupScale [kGroups]float32
		var groupMin [kGroups]float32
		var qs [kQK]uint32

		for g := 0; g < kGroups; g++ {
			gs := base + g*kGroup
			ge := gs + kGroup
			if gs >= n {
				groupScale[g] = 1
				continue
			}
			if ge > n {
				ge = n
			}
			if spec.hasDmin {
				mn, mx := minMaxRange(weights, gs, ge)
				sc := (mx - mn) / float32(spec.qMax)
				if sc == 0 {
					sc = 1
				}
				groupMin[g] = mn
				groupScale[g] = sc
				for j := 0; j < kGroup; j++ {
					i := gs + j
					q := 0
					if i < n {
						q = clampInt(roundToInt(float64((weights[i]-mn)/sc)), 0, spec.qMax)
					}
					qs[g*kGroup+j] = uint32(q)
				}
			} else {
				amax := maxAbsRange(weights, gs, ge)
				sc := amax / float32(spec.mid)
				if sc == 0 {
					sc = 1
				}
				groupScale[g] = sc
				for j := 0; j < kGroup; j++ {
					i := gs + j
					q := spec.mid
					if i < n {
						q = clampInt(roundToInt(float64(weights[i])/float64(sc))+spec.mid, 0, spec.qMax)
					}
					qs[g*kGroup+j] = uint32(q)
				}
			}
		}

		var d, dmin float32
		for g := 0; g < kGroups; g++ {
			if groupScale[g] > d {
				d = groupScale[g]
			}
			if spec.hasDmin {
				a := abs32(groupMin[g])
				if a > dmin {
					dmin = a
				}
			}
		}
		if d == 0 {
			d = 1
		}
		if spec.hasDmin && dmin == 0 {
			dmin = 1
		}

		putF32(raw[off:], d)
		putF32(raw[off+4:], dmin)
		for g := 0; g < kGroups; g++ {
			raw[off+8+g] = byte(clampInt(roundToInt(float64(groupScale[g])/float64(d)*255), 0, 255))
		}
		qsOff := off + 8 + kGroups
		if spec.hasDmin {
			for g := 0; g < kGroups; g++ {
				// Signed int8 bit pattern: min ≈ dmin * int8(u) / 127
				u := byte(0)
				if dmin > 0 {
					u = byte(clampInt(roundToInt(float64(groupMin[g])/float64(dmin)*127), -127, 127))
				}
				raw[qsOff+g] = u
			}
			qsOff += kGroups
		}
		bw := &bitWriter{buf: raw[qsOff : off+sbBytes]}
		for i := 0; i < kQK; i++ {
			bw.write(qs[i], spec.bits)
		}
	}

	return &Blob{
		Format:       format,
		Rows:         rows,
		Cols:         cols,
		Raw:          raw,
		BlockWeights: kQK,
	}, nil
}

func decodeKWeight(d, dmin float32, scaleU, minU byte, q uint32, spec kSpec) float32 {
	sc := d * float32(scaleU) / 255
	if sc == 0 {
		sc = d / 255
	}
	if spec.hasDmin {
		mn := dmin * float32(int8(minU)) / 127
		return mn + float32(q)*sc
	}
	return sc * float32(int(q)-spec.mid)
}

func forEachK(b *Blob, fn func(i int, w float32)) error {
	spec, ok := kSpecFor(b.Format)
	if !ok {
		return ErrUnsupported(b.Format, "kquant")
	}
	n := b.Rows * b.Cols
	sbBytes := kSuperBytes(spec)
	sbCount := len(b.Raw) / sbBytes
	for si := 0; si < sbCount; si++ {
		off := si * sbBytes
		d := getF32(b.Raw[off:])
		dmin := getF32(b.Raw[off+4:])
		scales := b.Raw[off+8 : off+8+kGroups]
		minsOff := off + 8 + kGroups
		qsOff := minsOff
		var mins []byte
		if spec.hasDmin {
			mins = b.Raw[minsOff : minsOff+kGroups]
			qsOff = minsOff + kGroups
		}
		br := &bitReader{buf: b.Raw[qsOff : off+sbBytes]}
		base := si * kQK
		for g := 0; g < kGroups; g++ {
			var minU byte
			if mins != nil {
				minU = mins[g]
			}
			for j := 0; j < kGroup; j++ {
				idx := base + g*kGroup + j
				q := br.read(spec.bits)
				if idx >= n {
					continue
				}
				fn(idx, decodeKWeight(d, dmin, scales[g], minU, q, spec))
			}
		}
	}
	return nil
}

func unpackK(b *Blob) ([]float32, error) {
	if b == nil || !b.Format.IsKQuant() {
		return nil, errFormat("UnpackK", b)
	}
	out := make([]float32, b.Rows*b.Cols)
	err := forEachK(b, func(i int, w float32) { out[i] = w })
	return out, err
}

func matVecK(b *Blob, x, y []float32) error {
	for i := range y[:b.Rows] {
		y[i] = 0
	}
	return forEachK(b, func(i int, w float32) {
		y[i/b.Cols] += w * x[i%b.Cols]
	})
}

func matVecTK(b *Blob, gy, gx []float32) error {
	return forEachK(b, func(i int, w float32) {
		gx[i%b.Cols] += w * gy[i/b.Cols]
	})
}
