package weights

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/openfluke/welvet/quant"
)

// EncodeBlobWire serializes a quant.Blob for checkpoints.
// FormatNone → raw LE float32 payload; block quants → EncodePackedWire.
func EncodeBlobWire(b *quant.Blob) []byte {
	if b == nil {
		return nil
	}
	if b.Format == quant.FormatNone {
		return append([]byte(nil), b.Raw...)
	}
	return EncodePackedWire(b)
}

// DecodeBlobWire reconstructs a quant.Blob (FormatNone or packed).
func DecodeBlobWire(format quant.Format, rows, cols int, wire []byte) (*quant.Blob, error) {
	return DecodePackedWire(format, rows, cols, wire)
}

// EncodePackedWire serializes a packed quant.Blob for ENTITY payload storage.
// Layout matches model/entity.EncodePackedBlob (ns, nm, nr, scales, mins, raw).
func EncodePackedWire(b *quant.Blob) []byte {
	if b == nil {
		return nil
	}
	ns := len(b.Scales)
	nm := len(b.Mins)
	nr := len(b.Raw)
	out := make([]byte, 12+ns*4+nm*4+nr)
	binary.LittleEndian.PutUint32(out[0:4], uint32(ns))
	binary.LittleEndian.PutUint32(out[4:8], uint32(nm))
	binary.LittleEndian.PutUint32(out[8:12], uint32(nr))
	off := 12
	for _, v := range b.Scales {
		binary.LittleEndian.PutUint32(out[off:off+4], math.Float32bits(v))
		off += 4
	}
	for _, v := range b.Mins {
		binary.LittleEndian.PutUint32(out[off:off+4], math.Float32bits(v))
		off += 4
	}
	copy(out[off:], b.Raw)
	return out
}

// DecodePackedWire reconstructs a quant.Blob from checkpoint wire bytes.
func DecodePackedWire(format quant.Format, rows, cols int, wire []byte) (*quant.Blob, error) {
	if format == quant.FormatNone {
		n := rows * cols
		need := n * 4
		if len(wire) < need {
			return nil, fmt.Errorf("weights: FormatNone wire short %d < %d", len(wire), need)
		}
		raw := make([]byte, need)
		copy(raw, wire[:need])
		return &quant.Blob{
			Format:       quant.FormatNone,
			Rows:         rows,
			Cols:         cols,
			Raw:          raw,
			BlockWeights: 1,
		}, nil
	}
	if len(wire) < 12 {
		return nil, fmt.Errorf("weights: packed wire too short")
	}
	ns := int(binary.LittleEndian.Uint32(wire[0:4]))
	nm := int(binary.LittleEndian.Uint32(wire[4:8]))
	nr := int(binary.LittleEndian.Uint32(wire[8:12]))
	need := 12 + ns*4 + nm*4 + nr
	if len(wire) < need {
		return nil, fmt.Errorf("weights: packed wire truncated (need %d, have %d)", need, len(wire))
	}
	off := 12
	scales := make([]float32, ns)
	for i := 0; i < ns; i++ {
		scales[i] = math.Float32frombits(binary.LittleEndian.Uint32(wire[off : off+4]))
		off += 4
	}
	mins := make([]float32, nm)
	for i := 0; i < nm; i++ {
		mins[i] = math.Float32frombits(binary.LittleEndian.Uint32(wire[off : off+4]))
		off += 4
	}
	raw := make([]byte, nr)
	copy(raw, wire[off:off+nr])
	b := &quant.Blob{
		Format: format,
		Rows:   rows,
		Cols:   cols,
		Raw:    raw,
		Scales: scales,
		Mins:   mins,
	}
	if format == quant.FormatQ4_0 {
		quant.EnsureQ4SIMDCache(b)
	}
	if format == quant.FormatBinaryPacked {
		quant.InferBinaryBlockWeights(b)
	}
	return b, nil
}
