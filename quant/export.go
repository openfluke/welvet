package quant

// ForEachK invokes fn for every weight in a k-quant Blob (streaming; no full unpack).
func ForEachK(b *Blob, fn func(i int, w float32)) error { return forEachK(b, fn) }

// ForEachIQ invokes fn for every weight in an IQ Blob (streaming; no full unpack).
func ForEachIQ(b *Blob, fn func(i int, w float32)) error { return forEachIQ(b, fn) }

// GetF32 reads a little-endian float32 from b[0:4].
func GetF32(b []byte) float32 { return getF32(b) }

// KSpec describes a k-quant layout for host/GPU decoders.
type KSpec struct {
	Bits    int
	HasDmin bool
	QMax    int
	Mid     int
	SBBytes int
}

// KSpecFor returns layout metadata for a k-quant format.
func KSpecFor(f Format) (KSpec, bool) {
	s, ok := kSpecFor(f)
	if !ok {
		return KSpec{}, false
	}
	return KSpec{Bits: s.bits, HasDmin: s.hasDmin, QMax: s.qMax, Mid: s.mid, SBBytes: kSuperBytes(s)}, true
}

// BitReader streams tightly packed n-bit codes.
type BitReader struct {
	br bitReader
}

// NewBitReader wraps raw packed bits.
func NewBitReader(buf []byte) *BitReader {
	return &BitReader{br: bitReader{buf: buf}}
}

// Read returns the next nBits as a uint32 code.
func (r *BitReader) Read(nBits int) uint32 {
	return r.br.read(nBits)
}
