//go:build amd64

package simd

//go:noescape
func q8BlockDot32Avx2(in *float32, qs *int8, scale float32) float64

//go:noescape
func q41BlockDot32Avx2(in *float32, packed4 *uint32, scale, min float32) float64

func dotQ8_0RowSimd(in []float32, scales []float32, qs []int8, baseW, n int, prev float64) float64 {
	sum := prev
	i := 0
	if baseW%32 == 0 {
		for i+32 <= n {
			block := (baseW + i) / 32
			sum += q8BlockDot32Avx2(&in[i], &qs[baseW+i], scales[block])
			i += 32
		}
	}
	if i < n {
		sum = dotQ8_0RowGo(in[i:], scales, qs, baseW+i, n-i, sum)
	}
	return sum
}

func dotQ8_0Rows4Simd(in []float32, scales []float32, qs []int8, baseW, n int, out []float32) {
	var acc [4]float64
	for i := 0; i < n; i += 32 {
		inBlk := &in[i]
		for r := 0; r < 4; r++ {
			bw := baseW + r*n + i
			acc[r] += q8BlockDot32Avx2(inBlk, &qs[bw], scales[bw/32])
		}
	}
	out[0] = float32(acc[0])
	out[1] = float32(acc[1])
	out[2] = float32(acc[2])
	out[3] = float32(acc[3])
}

func dotQ4_1RowSimd(in []float32, scales, mins []float32, packed []uint32, baseW, n int, prev float64) float64 {
	sum := prev
	i := 0
	if baseW%32 == 0 {
		for i+32 <= n {
			block := (baseW + i) / 32
			sum += q41BlockDot32Avx2(&in[i], &packed[(baseW+i)/8], scales[block], mins[block])
			i += 32
		}
	}
	if i < n {
		sum = dotQ4_1RowGo(in[i:], scales, mins, packed, baseW+i, n-i, sum)
	}
	return sum
}

func dotQ4_1Rows4Simd(in []float32, scales, mins []float32, packed []uint32, baseW, n int, out []float32) {
	var acc [4]float64
	for i := 0; i < n; i += 32 {
		inBlk := &in[i]
		for r := 0; r < 4; r++ {
			bw := baseW + r*n + i
			acc[r] += q41BlockDot32Avx2(inBlk, &packed[bw/8], scales[bw/32], mins[bw/32])
		}
	}
	out[0] = float32(acc[0])
	out[1] = float32(acc[1])
	out[2] = float32(acc[2])
	out[3] = float32(acc[3])
}
