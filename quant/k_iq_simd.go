package quant

import (
	"encoding/binary"
	"math"
)

// IQSIMDKind selects DotIQRow dequant mode (stored in Meta[2] after EnsureIQSIMDCache).
const (
	IQSIMDLinear  = 0 // scale * (q - mid)
	IQSIMDSigned1 = 1 // IQ1_S: scale * (±1)
	IQSIMDNonLin  = 2 // IQ4_NL grid
)

// EnsureKSIMDCache projects k-quant Raw into Int8QS + absolute per-group Scales/Mins.
// Group size is 16 (kGroup). Q6_K stores mid in Meta[0]; Meta[1]=1 when hasDmin.
func EnsureKSIMDCache(b *Blob) {
	if b == nil || !b.Format.IsKQuant() {
		return
	}
	spec, ok := kSpecFor(b.Format)
	if !ok {
		return
	}
	n := b.Rows * b.Cols
	if n <= 0 {
		return
	}
	sbBytes := kSuperBytes(spec)
	if sbBytes <= 0 || len(b.Raw) < sbBytes {
		return
	}
	sbCount := len(b.Raw) / sbBytes
	nGroups := sbCount * kGroups
	if len(b.Int8QS) >= n && len(b.Scales) >= nGroups {
		if !spec.hasDmin || len(b.Mins) >= nGroups {
			return
		}
	}
	if len(b.Raw) < sbCount*sbBytes {
		return
	}

	qs := make([]int8, sbCount*kQK)
	scales := make([]float32, nGroups)
	mins := make([]float32, nGroups)
	for si := 0; si < sbCount; si++ {
		off := si * sbBytes
		d := getF32(b.Raw[off:])
		dmin := getF32(b.Raw[off+4:])
		scU := b.Raw[off+8 : off+8+kGroups]
		minsOff := off + 8 + kGroups
		qsOff := minsOff
		var minU []byte
		if spec.hasDmin {
			minU = b.Raw[minsOff : minsOff+kGroups]
			qsOff = minsOff + kGroups
		}
		br := &bitReader{buf: b.Raw[qsOff : off+sbBytes]}
		gBase := si * kGroups
		qBase := si * kQK
		for g := 0; g < kGroups; g++ {
			sc := d * float32(scU[g]) / 255
			if sc == 0 {
				sc = d / 255
			}
			scales[gBase+g] = sc
			if minU != nil {
				mins[gBase+g] = dmin * float32(int8(minU[g])) / 127
			}
			for j := 0; j < kGroup; j++ {
				qs[qBase+g*kGroup+j] = int8(br.read(spec.bits))
			}
		}
	}
	b.Int8QS = qs[:n]
	b.Scales = scales
	if spec.hasDmin {
		b.Mins = mins
	} else {
		b.Mins = nil
	}
	b.Meta = []byte{byte(spec.mid), 0}
	if spec.hasDmin {
		b.Meta[1] = 1
	}
}

// KSIMDParams returns fused DotKRow parameters from a projected k blob.
func KSIMDParams(b *Blob) (hasDmin bool, mid int, ok bool) {
	if b == nil || !b.Format.IsKQuant() {
		return false, 0, false
	}
	spec, ok := kSpecFor(b.Format)
	if !ok {
		return false, 0, false
	}
	hasDmin = spec.hasDmin
	mid = spec.mid
	if len(b.Meta) >= 2 {
		mid = int(b.Meta[0])
		hasDmin = b.Meta[1] != 0
	}
	return hasDmin, mid, true
}

// EnsureIQSIMDCache expands bit-packed IQ codes into Int8QS; refreshes Meta for DotIQRow.
func EnsureIQSIMDCache(b *Blob) {
	if b == nil {
		return
	}
	spec, ok := iqSpecFor(b.Format)
	if !ok {
		return
	}
	if len(b.Meta) >= 2 {
		spec.bits = int(b.Meta[0])
		spec.scaleGroup = int(b.Meta[1])
	}
	if spec.bits <= 0 || spec.scaleGroup <= 0 {
		return
	}
	n := b.Rows * b.Cols
	if n <= 0 || len(b.Scales) == 0 {
		return
	}
	if len(b.Int8QS) >= n && len(b.Meta) >= 7 {
		return
	}
	padN := ((n + spec.scaleGroup - 1) / spec.scaleGroup) * spec.scaleGroup
	needBits := padN * spec.bits
	if len(b.Raw)*8 < needBits {
		return
	}
	qs := make([]int8, padN)
	br := &bitReader{buf: b.Raw}
	for i := 0; i < padN; i++ {
		qs[i] = int8(br.read(spec.bits))
	}
	b.Int8QS = qs[:n]

	kind := byte(IQSIMDLinear)
	if spec.bits == 1 {
		kind = byte(IQSIMDSigned1)
	} else if spec.nonlinear {
		kind = byte(IQSIMDNonLin)
	}
	meta := make([]byte, 7)
	meta[0] = byte(spec.bits)
	meta[1] = byte(spec.scaleGroup)
	meta[2] = kind
	binary.LittleEndian.PutUint32(meta[3:], math.Float32bits(spec.mid))
	b.Meta = meta
}

// IQSIMDParams returns DotIQRow parameters from a projected IQ blob.
func IQSIMDParams(b *Blob) (scaleGroup int, mid float32, kind int, ok bool) {
	if b == nil {
		return 0, 0, 0, false
	}
	spec, sok := iqSpecFor(b.Format)
	if !sok {
		return 0, 0, 0, false
	}
	scaleGroup = spec.scaleGroup
	mid = spec.mid
	kind = IQSIMDLinear
	if spec.bits == 1 {
		kind = IQSIMDSigned1
	} else if spec.nonlinear {
		kind = IQSIMDNonLin
	}
	if len(b.Meta) >= 2 {
		scaleGroup = int(b.Meta[1])
	}
	if len(b.Meta) >= 7 {
		kind = int(b.Meta[2])
		mid = math.Float32frombits(binary.LittleEndian.Uint32(b.Meta[3:7]))
	}
	if scaleGroup <= 0 {
		return 0, 0, 0, false
	}
	return scaleGroup, mid, kind, true
}

// EnsureAffineSIMDCache expands AffinePacked nibbles into Int8QS; validates Scales/Mins.
// Does not inflate F32Cache.
func EnsureAffineSIMDCache(b *Blob) {
	if b == nil || b.Format != FormatAffinePacked {
		return
	}
	InferAffineBlockWeights(b)
	rows, cols := b.Rows, b.Cols
	if rows <= 0 || cols <= 0 {
		return
	}
	bits := affineBits(b)
	group := b.BlockWeights
	if group <= 0 {
		group = AffineG64Group
	}
	if bits != 4 || cols%group != 0 {
		return
	}
	codesPerWord := 32 / bits
	if cols%codesPerWord != 0 {
		return
	}
	wordsPerRow := cols / codesPerWord
	groupsPerRow := cols / group
	needSB := rows * groupsPerRow
	if len(b.Scales) < needSB || len(b.Mins) < needSB {
		return
	}
	needRaw := rows * wordsPerRow * 4
	if len(b.Raw) < needRaw {
		return
	}
	n := rows * cols
	if len(b.Int8QS) >= n {
		return
	}
	qs := make([]int8, n)
	mask := uint32((1 << bits) - 1)
	for r := 0; r < rows; r++ {
		rowWords := r * wordsPerRow
		dst := r * cols
		for w := 0; w < wordsPerRow; w++ {
			word := getU32(b.Raw[(rowWords+w)*4:])
			base := dst + w*codesPerWord
			for j := 0; j < codesPerWord; j++ {
				qs[base+j] = int8((word >> uint(j*bits)) & mask)
			}
		}
	}
	b.Int8QS = qs
	if len(b.Meta) == 0 {
		b.Meta = []byte{byte(bits)}
	}
}
