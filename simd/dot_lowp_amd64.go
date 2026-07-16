//go:build amd64

package simd

import (
	"encoding/binary"
	"math"
)

//go:noescape
func cvtF16x8Avx(src *byte, dst *float32)

func dotF16PackedSimd(x []float32, w []byte, i0, n int, prev float64) float64 {
	sum := prev
	const tile = 8
	var buf [tile]float32
	for i := 0; i < n; {
		m := tile
		if n-i < m {
			m = n - i
		}
		if m == 8 {
			cvtF16x8Avx(&w[(i0+i)*2], &buf[0])
		} else {
			for k := 0; k < m; k++ {
				off := (i0 + i + k) * 2
				h := binary.LittleEndian.Uint16(w[off:])
				buf[k] = math.Float32frombits(f16bitsToF32bits(h))
			}
		}
		sum = DotTile(x[i:i+m], buf[:m], 0, m, sum)
		i += m
	}
	return sum
}

func dotFP8PackedSimd(x []float32, w []byte, i0, n, kind int, prev float64) float64 {
	return dotFP8PackedGo(x, w, i0, n, kind, prev)
}
