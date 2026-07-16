package quant

import (
	"fmt"
	"math"
)

func putF32(b []byte, v float32) {
	u := math.Float32bits(v)
	b[0] = byte(u)
	b[1] = byte(u >> 8)
	b[2] = byte(u >> 16)
	b[3] = byte(u >> 24)
}

func getF32(b []byte) float32 {
	u := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
	return math.Float32frombits(u)
}

func putU32(b []byte, v uint32) {
	b[0] = byte(v)
	b[1] = byte(v >> 8)
	b[2] = byte(v >> 16)
	b[3] = byte(v >> 24)
}

func getU32(b []byte) uint32 {
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

func absMean(w []float32, start, end int) float32 {
	if end <= start {
		return 0
	}
	var sum float32
	for i := start; i < end; i++ {
		sum += abs32(w[i])
	}
	return sum / float32(end-start)
}

func maxAbsRange(w []float32, start, end int) float32 {
	var m float32
	for i := start; i < end; i++ {
		a := abs32(w[i])
		if a > m {
			m = a
		}
	}
	return m
}

func minMaxRange(w []float32, start, end int) (float32, float32) {
	if start >= end {
		return 0, 0
	}
	mn, mx := w[start], w[start]
	for i := start + 1; i < end; i++ {
		if w[i] < mn {
			mn = w[i]
		}
		if w[i] > mx {
			mx = w[i]
		}
	}
	return mn, mx
}

func checkShape(op string, weights []float32, rows, cols int) error {
	if rows <= 0 || cols <= 0 || len(weights) < rows*cols {
		return errShape(op, rows, cols, len(weights))
	}
	return nil
}

func checkBlobYX(op string, b *Blob, x, y []float32) error {
	if b == nil {
		return fmt.Errorf("%s: nil blob", op)
	}
	if len(x) < b.Cols || len(y) < b.Rows {
		return fmt.Errorf("%s: size mismatch rows=%d cols=%d len(x)=%d len(y)=%d",
			op, b.Rows, b.Cols, len(x), len(y))
	}
	return nil
}

func checkBlobGYGX(op string, b *Blob, gy, gx []float32) error {
	if b == nil {
		return fmt.Errorf("%s: nil blob", op)
	}
	if len(gy) < b.Rows || len(gx) < b.Cols {
		return fmt.Errorf("%s: size mismatch rows=%d cols=%d len(gy)=%d len(gx)=%d",
			op, b.Rows, b.Cols, len(gy), len(gx))
	}
	return nil
}

// bitWriter packs little-endian bitfields into a byte slice.
type bitWriter struct {
	buf []byte
	pos int // bit offset
}

func newBitWriter(nBits int) *bitWriter {
	return &bitWriter{buf: make([]byte, (nBits+7)/8)}
}

func (bw *bitWriter) write(val uint32, nBits int) {
	for i := 0; i < nBits; i++ {
		if (val>>uint(i))&1 != 0 {
			bw.buf[bw.pos>>3] |= 1 << uint(bw.pos&7)
		}
		bw.pos++
	}
}

type bitReader struct {
	buf []byte
	pos int
}

func (br *bitReader) read(nBits int) uint32 {
	var v uint32
	for i := 0; i < nBits; i++ {
		if br.pos>>3 < len(br.buf) && (br.buf[br.pos>>3]>>uint(br.pos&7))&1 != 0 {
			v |= 1 << uint(i)
		}
		br.pos++
	}
	return v
}

func roundToInt(v float64) int {
	return int(math.Round(v))
}
