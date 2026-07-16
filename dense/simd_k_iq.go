package dense

import (
	"fmt"
	"math"

	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/simd"
)

// matVecKSIMD — group-wise decode + DotTile (still projects 16 f32/group; not full .s yet).
func matVecKSIMD(b *quant.Blob, x, y []float32) error {
	spec, ok := kSpecFromFormat(b.Format)
	if !ok {
		return fmt.Errorf("dense: not k-quant %s", b.Format)
	}
	rows, cols := b.Rows, b.Cols
	n := rows * cols
	sbBytes := kSuperBytes(spec)
	sbCount := len(b.Raw) / sbBytes
	for i := range y[:rows] {
		y[i] = 0
	}
	scratch := make([]float32, 16)

	for si := 0; si < sbCount; si++ {
		off := si * sbBytes
		raw := b.Raw[off : off+sbBytes]
		d := math.Float32frombits(uint32(raw[0]) | uint32(raw[1])<<8 | uint32(raw[2])<<16 | uint32(raw[3])<<24)
		dmin := math.Float32frombits(uint32(raw[4]) | uint32(raw[5])<<8 | uint32(raw[6])<<16 | uint32(raw[7])<<24)
		scales := raw[8 : 8+16]
		minsOff := 8 + 16
		qsOff := minsOff
		var mins []byte
		if spec.hasDmin {
			mins = raw[minsOff : minsOff+16]
			qsOff = minsOff + 16
		}
		br := bitAccum{buf: raw[qsOff:]}
		base := si * 256
		for g := 0; g < 16; g++ {
			var minU byte
			if mins != nil {
				minU = mins[g]
			}
			sc := d * float32(scales[g]) / 255
			if sc == 0 {
				sc = d / 255
			}
			var mn float32
			if spec.hasDmin {
				mn = dmin * float32(int8(minU)) / 127
			}
			for j := 0; j < 16; j++ {
				q := br.read(spec.bits)
				if spec.hasDmin {
					scratch[j] = mn + float32(q)*sc
				} else {
					scratch[j] = sc * float32(int(q)-spec.mid)
				}
			}
			start := base + g*16
			if start >= n {
				continue
			}
			nn := 16
			if start+nn > n {
				nn = n - start
			}
			r0 := start / cols
			c0 := start % cols
			if c0+nn <= cols {
				y[r0] += float32(simd.DotTile(x[c0:c0+nn], scratch[:nn], 0, nn, 0))
			} else {
				for j := 0; j < nn; j++ {
					i := start + j
					y[i/cols] += scratch[j] * x[i%cols]
				}
			}
		}
	}
	return nil
}

type kSpecLite struct {
	bits    int
	hasDmin bool
	mid     int
}

func kSpecFromFormat(f quant.Format) (kSpecLite, bool) {
	switch f {
	case quant.FormatQ2_K:
		return kSpecLite{bits: 2, hasDmin: true}, true
	case quant.FormatQ3_K:
		return kSpecLite{bits: 3, hasDmin: true}, true
	case quant.FormatQ4_K:
		return kSpecLite{bits: 4, hasDmin: true}, true
	case quant.FormatQ5_K:
		return kSpecLite{bits: 5, hasDmin: true}, true
	case quant.FormatQ6_K:
		return kSpecLite{bits: 6, hasDmin: false, mid: 32}, true
	default:
		return kSpecLite{}, false
	}
}

func kSuperBytes(spec kSpecLite) int {
	n := 8 + 16
	if spec.hasDmin {
		n += 16
	}
	n += (256*spec.bits + 7) / 8
	return n
}

type bitAccum struct {
	buf  []byte
	acc  uint64
	bits int
	p    int
}

func (b *bitAccum) read(n int) uint32 {
	for b.bits < n {
		if b.p >= len(b.buf) {
			return 0
		}
		b.acc |= uint64(b.buf[b.p]) << b.bits
		b.p++
		b.bits += 8
	}
	v := uint32(b.acc & ((1 << n) - 1))
	b.acc >>= n
	b.bits -= n
	return v
}

func matVecIQSIMD(b *quant.Blob, x, y []float32) error {
	bits, scaleGroup, nonlinear, mid, ok := iqParams(b.Format)
	if !ok {
		return fmt.Errorf("dense: not IQ %s", b.Format)
	}
	rows, cols := b.Rows, b.Cols
	n := rows * cols
	for i := range y[:rows] {
		y[i] = 0
	}
	if len(b.Scales) == 0 {
		return fmt.Errorf("dense: IQ missing scales")
	}
	scratch := make([]float32, scaleGroup)
	br := bitAccum{buf: b.Raw}
	groups := (n + scaleGroup - 1) / scaleGroup
	for gi := 0; gi < groups; gi++ {
		sc := float32(1)
		if gi < len(b.Scales) {
			sc = b.Scales[gi]
		}
		start := gi * scaleGroup
		nn := scaleGroup
		if start+nn > n {
			nn = n - start
		}
		for j := 0; j < nn; j++ {
			q := br.read(bits)
			if nonlinear {
				scratch[j] = iq4nl[q&15] * sc
			} else if bits == 1 {
				if q&1 == 0 {
					scratch[j] = -sc
				} else {
					scratch[j] = sc
				}
			} else {
				scratch[j] = (float32(q) - mid) * sc
			}
		}
		r0 := start / cols
		c0 := start % cols
		if c0+nn <= cols {
			y[r0] += float32(simd.DotTile(x[c0:c0+nn], scratch[:nn], 0, nn, 0))
		} else {
			for j := 0; j < nn; j++ {
				i := start + j
				y[i/cols] += scratch[j] * x[i%cols]
			}
		}
	}
	return nil
}

func iqParams(f quant.Format) (bits, scaleGroup int, nonlinear bool, mid float32, ok bool) {
	switch f {
	case quant.FormatIQ1_S:
		return 1, 32, false, 0.5, true
	case quant.FormatIQ2_XXS:
		return 2, 32, false, 1.5, true
	case quant.FormatIQ2_XS:
		return 2, 16, false, 1.5, true
	case quant.FormatIQ3_XXS:
		return 3, 32, false, 3.5, true
	case quant.FormatIQ3_S:
		return 3, 16, false, 3.5, true
	case quant.FormatIQ4_NL:
		return 4, 32, true, 0, true
	case quant.FormatIQ4_XS:
		return 4, 16, false, 7.5, true
	default:
		return 0, 0, false, 0, false
	}
}

var iq4nl = [16]float32{
	-1.0, -0.6961928, -0.52507305, -0.3949175,
	-0.28444138, -0.18477343, -0.091050036, 0.0,
	0.0795803, 0.1609302, 0.2461123, 0.33791524,
	0.44070983, 0.562617, 0.72295684, 1.0,
}
