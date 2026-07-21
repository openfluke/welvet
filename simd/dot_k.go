package simd

// IQ4NL grid (matches quant/iq.go) for DotIQRow nonlinear mode.
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

// DotKRow: fused k-quant row — groups of 16 with absolute scales/mins.
// hasDmin: y += mn*Σx + sc*Σ(x*q); else y += sc*Σ(x*(q-mid)).
func DotKRow(in, scales, mins []float32, qs []int8, baseW, n int, hasDmin bool, mid int, prev float64) float64 {
	const gsz = 16
	if n <= 0 || len(in) < n || baseW < 0 || baseW+n > len(qs) {
		return prev
	}
	last := (baseW + n - 1) / gsz
	if last >= len(scales) {
		return prev
	}
	if hasDmin && last >= len(mins) {
		return prev
	}
	sum := prev
	i := 0
	if baseW%gsz == 0 {
		for i+gsz <= n {
			g := (baseW + i) / gsz
			sc := float64(scales[g])
			off := baseW + i
			if hasDmin {
				mn := float64(mins[g])
				accQ := 0.0
				sumX := 0.0
				for j := 0; j < gsz; j++ {
					xi := float64(in[i+j])
					accQ += xi * float64(uint8(qs[off+j]))
					sumX += xi
				}
				sum += accQ*sc + mn*sumX
			} else {
				acc := 0.0
				for j := 0; j < gsz; j++ {
					acc += float64(in[i+j]) * float64(int(uint8(qs[off+j]))-mid)
				}
				sum += acc * sc
			}
			i += gsz
		}
	}
	for ; i < n; i++ {
		g := (baseW + i) / gsz
		q := int(uint8(qs[baseW+i]))
		if hasDmin {
			sum += float64(in[i]) * (float64(mins[g]) + float64(q)*float64(scales[g]))
		} else {
			sum += float64(in[i]) * float64(scales[g]) * float64(q-mid)
		}
	}
	return sum
}

// DotIQRow: fused IQ row — scaleGroup codes share one scale.
// kind: 0=linear (q-mid), 1=IQ1 ±1, 2=IQ4_NL grid.
func DotIQRow(in, scales []float32, qs []int8, baseW, n, scaleGroup int, mid float32, kind int, prev float64) float64 {
	if n <= 0 || len(in) < n || baseW < 0 || baseW+n > len(qs) || scaleGroup <= 0 {
		return prev
	}
	last := (baseW + n - 1) / scaleGroup
	if last >= len(scales) {
		return prev
	}
	sum := prev
	i := 0
	if baseW%scaleGroup == 0 {
		for i+scaleGroup <= n {
			sc := float64(scales[(baseW+i)/scaleGroup])
			off := baseW + i
			acc := 0.0
			switch kind {
			case 1: // IQ1_S
				for j := 0; j < scaleGroup; j++ {
					q := uint8(qs[off+j]) & 1
					code := -1.0
					if q != 0 {
						code = 1.0
					}
					acc += float64(in[i+j]) * code
				}
			case 2: // IQ4_NL
				for j := 0; j < scaleGroup; j++ {
					acc += float64(in[i+j]) * float64(iq4nlGrid[uint8(qs[off+j])&15])
				}
			default: // linear mid
				m := float64(mid)
				for j := 0; j < scaleGroup; j++ {
					acc += float64(in[i+j]) * (float64(uint8(qs[off+j])) - m)
				}
			}
			sum += acc * sc
			i += scaleGroup
		}
	}
	for ; i < n; i++ {
		sc := float64(scales[(baseW+i)/scaleGroup])
		q := uint8(qs[baseW+i])
		var code float64
		switch kind {
		case 1:
			if q&1 != 0 {
				code = 1
			} else {
				code = -1
			}
		case 2:
			code = float64(iq4nlGrid[q&15])
		default:
			code = float64(q) - float64(mid)
		}
		sum += float64(in[i]) * code * sc
	}
	return sum
}

// DotAffineRow: w = s·code + β over groups; qs are expanded 4-bit codes.
// sumX[g] is Σx over group g (shared across rows); scaleBase indexes Scales/Mins for this row.
func DotAffineRow(in, scales, mins, sumX []float32, qs []int8, baseW, n, group, scaleBase int, prev float64) float64 {
	if n <= 0 || group <= 0 || len(in) < n || baseW < 0 || baseW+n > len(qs) {
		return prev
	}
	if n%group != 0 {
		return prev
	}
	groups := n / group
	if scaleBase+groups > len(scales) || scaleBase+groups > len(mins) || len(sumX) < groups {
		return prev
	}
	sum := prev
	for g := 0; g < groups; g++ {
		sc := float64(scales[scaleBase+g])
		beta := float64(mins[scaleBase+g])
		off := baseW + g*group
		accQ := 0.0
		base := g * group
		// Prefer int8 tile MAC for the code·x path when both sides can be treated as int8 — activations stay f32.
		for j := 0; j < group; j++ {
			accQ += float64(in[base+j]) * float64(uint8(qs[off+j]))
		}
		sum += accQ*sc + beta*float64(sumX[g])
	}
	return sum
}

// AffineSumX fills sumX[g] = Σ x[g*group : (g+1)*group].
func AffineSumX(x, sumX []float32, n, group int) {
	if group <= 0 || n%group != 0 || len(sumX) < n/group {
		return
	}
	groups := n / group
	for g := 0; g < groups; g++ {
		var s float32
		base := g * group
		for j := 0; j < group; j++ {
			s += x[base+j]
		}
		sumX[g] = s
	}
}
