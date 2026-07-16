package quant

import "math"

// Q4Packed holds u32 SIMD words (4 per 32-weight block). Built once at pack time.
// Scales on Blob is reused for per-block f32 scales when FormatQ4_0.
func EnsureQ4SIMDCache(b *Blob) {
	if b == nil || b.Format != FormatQ4_0 || len(b.Raw) < 20 {
		return
	}
	const bw = 32
	n := b.Rows * b.Cols
	blocks := (n + bw - 1) / bw
	if len(b.Raw) < blocks*20 {
		return
	}
	if len(b.Scales) >= blocks && len(b.Q4Packed) >= blocks*4 {
		return
	}
	scales := make([]float32, blocks)
	packed := make([]uint32, blocks*4)
	for bi := 0; bi < blocks; bi++ {
		off := bi * 20
		scales[bi] = math.Float32frombits(uint32(b.Raw[off]) | uint32(b.Raw[off+1])<<8 |
			uint32(b.Raw[off+2])<<16 | uint32(b.Raw[off+3])<<24)
		for k := 0; k < 4; k++ {
			packed[bi*4+k] = uint32(b.Raw[off+4+k*4]) |
				uint32(b.Raw[off+5+k*4])<<8 |
				uint32(b.Raw[off+6+k*4])<<16 |
				uint32(b.Raw[off+7+k*4])<<24
		}
	}
	b.Scales = scales
	b.Q4Packed = packed
}

// Q4SIMD returns cached SIMD views for Q4_0 blobs.
func Q4SIMD(b *Blob) (scales []float32, packed []uint32, ok bool) {
	if b == nil || b.Format != FormatQ4_0 {
		return nil, nil, false
	}
	EnsureQ4SIMDCache(b)
	if len(b.Scales) == 0 || len(b.Q4Packed) == 0 {
		return nil, nil, false
	}
	return b.Scales, b.Q4Packed, true
}
