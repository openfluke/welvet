package weights

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
)

// ConvertOpts selects the destination numerical type and/or packed quant format.
//
// Storage truth after Convert is only DType + Format (plus Native or Packed
// payload). There is no retained parallel “master” weight tensor — conversion
// decodes the current payload to a temporary f32 scratch, then re-encodes.
// Low-bit / quant hops are therefore lossy.
//
//	FormatNone + DType X  → FormatNone storage in dtype X (Native, or f32 buffer)
//	Format Q* / Binary…   → Packed quant blob (DType recorded as Float32)
type ConvertOpts struct {
	DType  core.DType   // FormatNone element type (ignored when Format != none)
	Format quant.Format // FormatNone = native dtype storage
}

// Convert re-encodes s in place to the requested dtype/format.
func Convert(s *Store, opts ConvertOpts) error {
	if s == nil {
		return fmt.Errorf("weights: Convert nil store")
	}
	if opts.Format != quant.FormatNone && !quant.Supported(opts.Format) {
		return fmt.Errorf("weights: unsupported format %s", opts.Format.String())
	}

	// Decode current storage → scratch (not retained as truth).
	scratch, err := s.FlattenF32()
	if err != nil {
		return err
	}

	// Drop every payload; re-encode solely into the destination form.
	s.Packed = nil
	s.Native = nil
	s.masterF32 = nil
	s.gpuF32 = nil
	s.wireF64 = nil
	s.Scale = 1

	if opts.Format == quant.FormatNone {
		return encodeFormatNone(s, opts.DType, scratch)
	}
	return encodePacked(s, opts.Format, scratch)
}

// Converted returns a new Store with the requested dtype/format (s unchanged).
func Converted(s *Store, opts ConvertOpts) (*Store, error) {
	if s == nil {
		return nil, fmt.Errorf("weights: Converted nil store")
	}
	scratch, err := s.FlattenF32()
	if err != nil {
		return nil, err
	}
	dst := &Store{Rows: s.Rows, Cols: s.Cols, Scale: 1}
	if len(s.Bias) > 0 {
		dst.Bias = append([]float64(nil), s.Bias...)
	}
	if opts.Format == quant.FormatNone {
		if err := encodeFormatNone(dst, opts.DType, scratch); err != nil {
			return nil, err
		}
		return dst, nil
	}
	if err := encodePacked(dst, opts.Format, scratch); err != nil {
		return nil, err
	}
	return dst, nil
}

func encodeFormatNone(s *Store, dt core.DType, scratch []float32) error {
	n := s.Rows * s.Cols
	if len(scratch) < n {
		return fmt.Errorf("weights: scratch len %d < %d", len(scratch), n)
	}
	s.Format = quant.FormatNone
	s.DType = dt
	s.Packed = nil
	if dt == core.DTypeFloat32 {
		// FormatNone+F32 storage slot (the f32 buffer IS the payload).
		s.masterF32 = append([]float32(nil), scratch[:n]...)
		s.Native = nil
		s.Scale = 1
		return nil
	}
	raw, scale, err := packNative(dt, scratch[:n])
	if err != nil {
		return err
	}
	s.Native = raw
	s.Scale = scale
	s.masterF32 = nil
	return nil
}

func encodePacked(s *Store, format quant.Format, scratch []float32) error {
	n := s.Rows * s.Cols
	if len(scratch) < n {
		return fmt.Errorf("weights: scratch len %d < %d", len(scratch), n)
	}
	b, err := quant.Pack(format, scratch[:n], s.Rows, s.Cols)
	if err != nil {
		return err
	}
	s.Format = format
	s.DType = core.DTypeFloat32 // pack formats are f32-defined; payload is Packed
	s.Packed = b
	s.Native = nil
	s.masterF32 = nil
	s.Scale = 1
	s.gpuF32 = nil
	s.wireF64 = nil
	if format == quant.FormatQ4_0 {
		quant.EnsureQ4SIMDCache(b)
	}
	return nil
}
