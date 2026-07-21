package quant

import "math"

// EnsureQ8SIMDCache builds scales[] + Int8QS[] once (fused DotQ8_0Row layout).
func EnsureQ8SIMDCache(b *Blob) {
	if b == nil || b.Format != FormatQ8_0 {
		return
	}
	const bw, bb = 32, 36
	n := b.Rows * b.Cols
	blocks := (n + bw - 1) / bw
	if len(b.Raw) < blocks*bb {
		return
	}
	if len(b.Scales) >= blocks && len(b.Int8QS) >= n {
		return
	}
	scales := make([]float32, blocks)
	qs := make([]int8, blocks*bw)
	for bi := 0; bi < blocks; bi++ {
		off := bi * bb
		scales[bi] = math.Float32frombits(uint32(b.Raw[off]) | uint32(b.Raw[off+1])<<8 |
			uint32(b.Raw[off+2])<<16 | uint32(b.Raw[off+3])<<24)
		for j := 0; j < bw; j++ {
			qs[bi*bw+j] = int8(b.Raw[off+4+j])
		}
	}
	b.Scales = scales
	b.Int8QS = qs[:n]
}

// EnsureQ41SIMDCache builds scales/mins + nibble packed[] like Q4_0 GPU words.
func EnsureQ41SIMDCache(b *Blob) {
	if b == nil || b.Format != FormatQ4_1 {
		return
	}
	const bw, bb = 32, 24
	n := b.Rows * b.Cols
	blocks := (n + bw - 1) / bw
	if len(b.Raw) < blocks*bb {
		return
	}
	if len(b.Scales) >= blocks && len(b.Mins) >= blocks && len(b.Q4Packed) >= blocks*4 {
		return
	}
	scales := make([]float32, blocks)
	mins := make([]float32, blocks)
	packed := make([]uint32, blocks*4)
	for bi := 0; bi < blocks; bi++ {
		off := bi * bb
		scales[bi] = math.Float32frombits(uint32(b.Raw[off]) | uint32(b.Raw[off+1])<<8 |
			uint32(b.Raw[off+2])<<16 | uint32(b.Raw[off+3])<<24)
		mins[bi] = math.Float32frombits(uint32(b.Raw[off+4]) | uint32(b.Raw[off+5])<<8 |
			uint32(b.Raw[off+6])<<16 | uint32(b.Raw[off+7])<<24)
		for k := 0; k < 4; k++ {
			base := off + 8 + k*4
			packed[bi*4+k] = uint32(b.Raw[base]) | uint32(b.Raw[base+1])<<8 |
				uint32(b.Raw[base+2])<<16 | uint32(b.Raw[base+3])<<24
		}
	}
	b.Scales = scales
	b.Mins = mins
	b.Q4Packed = packed
}

// EnsureQ5_0SIMDCache expands 5-bit codes to signed int8 (q-16) once; reuse DotQ8 kernel.
func EnsureQ5_0SIMDCache(b *Blob) {
	if b == nil || b.Format != FormatQ5_0 {
		return
	}
	const bw, bb = 32, 24
	n := b.Rows * b.Cols
	blocks := (n + bw - 1) / bw
	if len(b.Raw) < blocks*bb {
		return
	}
	if len(b.Scales) >= blocks && len(b.Int8QS) >= n {
		return
	}
	scales := make([]float32, blocks)
	qs := make([]int8, blocks*bw)
	for bi := 0; bi < blocks; bi++ {
		off := bi * bb
		scales[bi] = math.Float32frombits(uint32(b.Raw[off]) | uint32(b.Raw[off+1])<<8 |
			uint32(b.Raw[off+2])<<16 | uint32(b.Raw[off+3])<<24)
		var acc uint64
		bits := 0
		p := off + 4
		for j := 0; j < bw; j++ {
			for bits < 5 {
				acc |= uint64(b.Raw[p]) << bits
				p++
				bits += 8
			}
			q := int(acc & 31)
			acc >>= 5
			bits -= 5
			qs[bi*bw+j] = int8(q - 16)
		}
	}
	b.Scales = scales
	b.Int8QS = qs[:n]
}

// EnsureQ5_1SIMDCache expands to uint8 qs in Int8QS + scales/mins.
func EnsureQ5_1SIMDCache(b *Blob) {
	if b == nil || b.Format != FormatQ5_1 {
		return
	}
	const bw, bb = 32, 28
	n := b.Rows * b.Cols
	blocks := (n + bw - 1) / bw
	if len(b.Raw) < blocks*bb {
		return
	}
	if len(b.Scales) >= blocks && len(b.Mins) >= blocks && len(b.Int8QS) >= n {
		return
	}
	scales := make([]float32, blocks)
	mins := make([]float32, blocks)
	qs := make([]int8, blocks*bw)
	for bi := 0; bi < blocks; bi++ {
		off := bi * bb
		scales[bi] = math.Float32frombits(uint32(b.Raw[off]) | uint32(b.Raw[off+1])<<8 |
			uint32(b.Raw[off+2])<<16 | uint32(b.Raw[off+3])<<24)
		mins[bi] = math.Float32frombits(uint32(b.Raw[off+4]) | uint32(b.Raw[off+5])<<8 |
			uint32(b.Raw[off+6])<<16 | uint32(b.Raw[off+7])<<24)
		var acc uint64
		bits := 0
		p := off + 8
		for j := 0; j < bw; j++ {
			for bits < 5 {
				acc |= uint64(b.Raw[p]) << bits
				p++
				bits += 8
			}
			q := byte(acc & 31)
			acc >>= 5
			bits -= 5
			qs[bi*bw+j] = int8(q)
		}
	}
	b.Scales = scales
	b.Mins = mins
	b.Int8QS = qs[:n]
}

// EnsureFusedSIMDCache projects any classic/k format used by simd_fuse.
func EnsureFusedSIMDCache(b *Blob) {
	if b == nil {
		return
	}
	switch b.Format {
	case FormatQ4_0:
		EnsureQ4SIMDCache(b)
	case FormatQ8_0:
		EnsureQ8SIMDCache(b)
	case FormatQ4_1:
		EnsureQ41SIMDCache(b)
	case FormatQ5_0:
		EnsureQ5_0SIMDCache(b)
	case FormatQ5_1:
		EnsureQ5_1SIMDCache(b)
	case FormatQ2_K, FormatQ3_K, FormatQ4_K, FormatQ5_K, FormatQ6_K:
		EnsureKSIMDCache(b)
	case FormatIQ1_S, FormatIQ2_XXS, FormatIQ2_XS, FormatIQ3_XXS, FormatIQ3_S,
		FormatIQ4_NL, FormatIQ4_XS:
		EnsureIQSIMDCache(b)
	case FormatAffinePacked:
		EnsureAffineSIMDCache(b)
	case FormatTernaryPacked, FormatBinaryPacked:
		EnsureFloatCache(b)
	}
}

// EnsureFloatCache unpacks Ternary/Binary weights to FP32 once for parallel DotTile GEMV.
// Keeps packed Raw on disk/RAM; F32Cache is the simd_fuse compute view.
// Skips BinaryG128 (Bonsai) — full inflate of 27B would OOM; use native matvec.
// k/IQ/Affine use Ensure*SIMDCache (Int8QS) instead — do not call this for those formats.
func EnsureFloatCache(b *Blob) {
	if b == nil {
		return
	}
	if isBinaryG128(b) {
		return
	}
	n := b.Rows * b.Cols
	if n <= 0 || len(b.F32Cache) >= n {
		return
	}
	// Refuse huge inflates (>512MiB) — force native packed path.
	if n > 128<<20 {
		return
	}
	all, err := Unpack(b)
	if err != nil || len(all) < n {
		return
	}
	b.F32Cache = all
}
