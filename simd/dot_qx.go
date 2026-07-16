package simd

// DotQ8_0Row: prev + sum_i in[i] * (q[i] * scale_block).
// qs is contiguous int8 (one per weight), scales[block] for each 32-weight block.
// Fused: does not expand a full FP32 weight row.
func DotQ8_0Row(in []float32, scales []float32, qs []int8, baseW, n int, prev float64) float64 {
	if n <= 0 || len(in) < n || baseW < 0 {
		return prev
	}
	if baseW+n > len(qs) {
		return prev
	}
	last := (baseW + n - 1) / 32
	if last >= len(scales) {
		return prev
	}
	if simdEnabled() {
		return dotQ8_0RowSimd(in, scales, qs, baseW, n, prev)
	}
	return dotQ8_0RowGo(in, scales, qs, baseW, n, prev)
}

// DotQ8_0Rows4 writes 4 consecutive row dots (activations stay hot).
func DotQ8_0Rows4(in []float32, scales []float32, qs []int8, baseW, n int, out []float32) {
	if len(out) < 4 || n <= 0 || len(in) < n {
		return
	}
	if simdEnabled() && baseW%32 == 0 && n%32 == 0 {
		dotQ8_0Rows4Simd(in, scales, qs, baseW, n, out)
		return
	}
	for r := 0; r < 4; r++ {
		out[r] = float32(DotQ8_0Row(in, scales, qs, baseW+r*n, n, 0))
	}
}

func dotQ8_0RowGo(in []float32, scales []float32, qs []int8, baseW, n int, prev float64) float64 {
	sum := prev
	i := 0
	if baseW%32 == 0 {
		for i+32 <= n {
			sc := float64(scales[(baseW+i)/32])
			acc := 0.0
			off := baseW + i
			for j := 0; j < 32; j++ {
				acc += float64(in[i+j]) * float64(qs[off+j])
			}
			sum += acc * sc
			i += 32
		}
	}
	for ; i < n; i++ {
		sc := float64(scales[(baseW+i)/32])
		sum += float64(in[i]) * float64(qs[baseW+i]) * sc
	}
	return sum
}

// DotQ4_1Row: prev + sum_i in[i] * (min + q[i]*scale) with q∈[0,15].
// packed is 4×u32 nibbles per 32-weight block (same word layout as Q4_0 GPU pack).
func DotQ4_1Row(in []float32, scales, mins []float32, packed []uint32, baseW, n int, prev float64) float64 {
	if n <= 0 || len(in) < n || baseW < 0 {
		return prev
	}
	if (baseW+n+7)/8 > len(packed) {
		return prev
	}
	last := (baseW + n - 1) / 32
	if last >= len(scales) || last >= len(mins) {
		return prev
	}
	if simdEnabled() {
		return dotQ4_1RowSimd(in, scales, mins, packed, baseW, n, prev)
	}
	return dotQ4_1RowGo(in, scales, mins, packed, baseW, n, prev)
}

func DotQ4_1Rows4(in []float32, scales, mins []float32, packed []uint32, baseW, n int, out []float32) {
	if len(out) < 4 || n <= 0 || len(in) < n {
		return
	}
	if simdEnabled() && baseW%32 == 0 && n%32 == 0 {
		dotQ4_1Rows4Simd(in, scales, mins, packed, baseW, n, out)
		return
	}
	for r := 0; r < 4; r++ {
		out[r] = float32(DotQ4_1Row(in, scales, mins, packed, baseW+r*n, n, 0))
	}
}

func dotQ4_1RowGo(in []float32, scales, mins []float32, packed []uint32, baseW, n int, prev float64) float64 {
	sum := prev
	i := 0
	if baseW%32 == 0 {
		for i+32 <= n {
			block := (baseW + i) / 32
			sc := float64(scales[block])
			mn := float64(mins[block])
			accQ := 0.0
			sumX := 0.0
			pkOff := (baseW + i) / 8
			for w := 0; w < 4; w++ {
				word := packed[pkOff+w]
				for nib := 0; nib < 8; nib++ {
					q := float64((word >> (uint(nib) * 4)) & 0xF)
					xi := float64(in[i+w*8+nib])
					accQ += xi * q
					sumX += xi
				}
			}
			sum += accQ*sc + mn*sumX
			i += 32
		}
	}
	for ; i < n; i++ {
		gi := baseW + i
		block := gi / 32
		q := float64((packed[gi/8] >> uint((gi%8)*4)) & 0xF)
		sum += float64(in[i]) * (float64(mins[block]) + q*float64(scales[block]))
	}
	return sum
}

// DotQ5_0Row uses int8 qs (q_stored-16) + scales — projected once via EnsureQ5SIMDCache.
func DotQ5_0Row(in []float32, scales []float32, qs []int8, baseW, n int, prev float64) float64 {
	return DotQ8_0Row(in, scales, qs, baseW, n, prev)
}

func DotQ5_0Rows4(in []float32, scales []float32, qs []int8, baseW, n int, out []float32) {
	DotQ8_0Rows4(in, scales, qs, baseW, n, out)
}

// DotQ5_1Row: asymmetric — qs are raw 0..31, plus mins.
func DotQ5_1Row(in []float32, scales, mins []float32, qs []int8, baseW, n int, prev float64) float64 {
	if n <= 0 || len(in) < n || baseW < 0 || baseW+n > len(qs) {
		return prev
	}
	last := (baseW + n - 1) / 32
	if last >= len(scales) || last >= len(mins) {
		return prev
	}
	sum := prev
	i := 0
	if baseW%32 == 0 {
		for i+32 <= n {
			block := (baseW + i) / 32
			sc := float64(scales[block])
			mn := float64(mins[block])
			accQ := 0.0
			sumX := 0.0
			off := baseW + i
			for j := 0; j < 32; j++ {
				xi := float64(in[i+j])
				accQ += xi * float64(uint8(qs[off+j]))
				sumX += xi
			}
			sum += accQ*sc + mn*sumX
			i += 32
		}
	}
	for ; i < n; i++ {
		block := (baseW + i) / 32
		sum += float64(in[i]) * (float64(mins[block]) + float64(uint8(qs[baseW+i]))*float64(scales[block]))
	}
	return sum
}
