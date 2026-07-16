package weights

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
)

// Store holds native weight payloads. Runtime truth is DType + Format (no QAT morph).
//
// masterF32 is only an internal pack/init bridge for ggml-style block formats that
// are defined on f32 source weights — it is NOT the activation tensor type.
type Store struct {
	DType  core.DType
	Format quant.Format
	Rows   int
	Cols   int

	masterF32 []float32 // pack source / FormatNone+Float32 storage
	Native    []byte
	Scale     float32
	Packed    *quant.Blob

	// Bias in float64; applied via core.FromFloat64 into activation dtype T.
	Bias []float64

	gpuF32  []float32
	wireF64 []float64
}

// New builds FormatNone weights from any Numeric init slice + target dtype/format.
func New[T core.Numeric](rows, cols int, data []T, dt core.DType, format quant.Format) (*Store, error) {
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("weights: bad shape %dx%d", rows, cols)
	}
	n := rows * cols
	src := make([]float32, n)
	if data != nil {
		if len(data) < n {
			return nil, fmt.Errorf("weights: need %d values", n)
		}
		for i := 0; i < n; i++ {
			src[i] = float32(core.AsFloat64(data[i]))
		}
	}
	s := &Store{
		DType:     dt,
		Format:    quant.FormatNone,
		Rows:      rows,
		Cols:      cols,
		masterF32: src,
		Scale:     1,
	}
	if err := s.SetDType(dt); err != nil {
		return nil, err
	}
	if format != quant.FormatNone {
		if err := s.Pack(format); err != nil {
			return nil, err
		}
	}
	return s, nil
}

// SetDType packs master into FormatNone native storage for dt.
func (s *Store) SetDType(dt core.DType) error {
	if s == nil {
		return fmt.Errorf("weights: nil store")
	}
	if len(s.masterF32) < s.Rows*s.Cols {
		return fmt.Errorf("weights: missing master source")
	}
	s.DType = dt
	s.Format = quant.FormatNone
	s.Packed = nil
	s.gpuF32 = nil
	s.wireF64 = nil
	if dt == core.DTypeFloat32 {
		s.Native = nil
		s.Scale = 1
		return nil
	}
	raw, scale, err := packNative(dt, s.masterF32[:s.Rows*s.Cols])
	if err != nil {
		return err
	}
	s.Native = raw
	s.Scale = scale
	return nil
}

// Pack converts current weights into a QuantFormat blob.
func (s *Store) Pack(format quant.Format) error {
	if s == nil {
		return quant.ErrUnsupported(format, "Pack")
	}
	src, err := s.float32View()
	if err != nil {
		return err
	}
	if format == quant.FormatNone {
		s.Format = quant.FormatNone
		s.Packed = nil
		s.gpuF32 = nil
		s.wireF64 = nil
		return nil
	}
	b, err := quant.Pack(format, src, s.Rows, s.Cols)
	if err != nil {
		return err
	}
	s.Format = format
	s.Packed = b
	s.gpuF32 = nil
	s.wireF64 = nil
	return nil
}

func (s *Store) float32View() ([]float32, error) {
	if s == nil {
		return nil, fmt.Errorf("weights: nil")
	}
	n := s.Rows * s.Cols
	if s.Format != quant.FormatNone {
		if s.Packed == nil {
			return nil, fmt.Errorf("weights: packed format %s missing blob", s.Format)
		}
		return quant.Unpack(s.Packed)
	}
	if s.DType == core.DTypeFloat32 {
		if len(s.masterF32) < n {
			return nil, fmt.Errorf("weights: master len")
		}
		return s.masterF32[:n], nil
	}
	if len(s.Native) == 0 {
		if len(s.masterF32) >= n {
			return s.masterF32[:n], nil
		}
		return nil, fmt.Errorf("weights: no native payload for %s", s.DType)
	}
	return unpackNative(s.DType, s.Native, s.Scale, n)
}

// MatVec computes y = W @ x for any Numeric activation dtype.
// Accumulation is float64 — not hardcoded to float32.
func MatVec[T core.Numeric](s *Store, x, y []T) error {
	if s == nil {
		return fmt.Errorf("weights: nil")
	}
	xf := core.SliceAsFloat64(x)
	yf := make([]float64, len(y))
	if err := s.matVecF64(xf, yf); err != nil {
		return err
	}
	core.SliceFromFloat64(yf, y)
	return nil
}

// MatVecT computes gx += W^T @ gy for any Numeric activation dtype (float64 acc).
func MatVecT[T core.Numeric](s *Store, gy, gx []T) error {
	if s == nil {
		return fmt.Errorf("weights: nil")
	}
	gyf := core.SliceAsFloat64(gy)
	gxf := core.SliceAsFloat64(gx)
	if err := s.matVecTF64(gyf, gxf); err != nil {
		return err
	}
	core.SliceFromFloat64(gxf, gx)
	return nil
}

func (s *Store) matVecF64(x, y []float64) error {
	if s.Format != quant.FormatNone {
		if s.Packed == nil {
			return fmt.Errorf("weights: nil packed blob")
		}
		// Block quants: MatVec in f32 then promote (ggml layout is f32-defined).
		xf := make([]float32, len(x))
		for i := range x {
			xf[i] = float32(x[i])
		}
		yf := make([]float32, len(y))
		if err := quant.MatVec(s.Packed, xf, yf); err != nil {
			return err
		}
		for i := range y {
			y[i] = float64(yf[i])
		}
		return nil
	}
	return matVecNativeF64(s, x, y)
}

func (s *Store) matVecTF64(gy, gx []float64) error {
	if s.Format != quant.FormatNone {
		if s.Packed == nil {
			return fmt.Errorf("weights: nil packed blob")
		}
		gyf := make([]float32, len(gy))
		gxf := make([]float32, len(gx))
		for i := range gy {
			gyf[i] = float32(gy[i])
		}
		for i := range gx {
			gxf[i] = float32(gx[i])
		}
		if err := quant.MatVecT(s.Packed, gyf, gxf); err != nil {
			return err
		}
		for i := range gx {
			gx[i] = float64(gxf[i])
		}
		return nil
	}
	return matVecTNativeF64(s, gy, gx)
}

func (s *Store) matVecF32(x, y []float32) error {
	xf := make([]float64, len(x))
	yf := make([]float64, len(y))
	for i := range x {
		xf[i] = float64(x[i])
	}
	if err := s.matVecF64(xf, yf); err != nil {
		return err
	}
	for i := range y {
		y[i] = float32(yf[i])
	}
	return nil
}

func (s *Store) matVecTF32(gy, gx []float32) error {
	gyf := make([]float64, len(gy))
	gxf := make([]float64, len(gx))
	for i := range gy {
		gyf[i] = float64(gy[i])
	}
	for i := range gx {
		gxf[i] = float64(gx[i])
	}
	if err := s.matVecTF64(gyf, gxf); err != nil {
		return err
	}
	for i := range gx {
		gx[i] = float32(gxf[i])
	}
	return nil
}

// GPUWireF32 returns the f32 compute-wire buffer for WebGPU / SIMD DotTile paths.
// FormatNone: stream-decode native storage. Block quants: Unpack packed blob once.
// Storage remains native/packed; this is the kernel wire only (same class as FormatNone DotTile).
func (s *Store) GPUWireF32() ([]float32, error) {
	if s == nil {
		return nil, fmt.Errorf("weights: nil")
	}
	if s.gpuF32 != nil {
		return s.gpuF32, nil
	}
	n := s.Rows * s.Cols
	if s.Format != quant.FormatNone {
		if s.Packed == nil {
			return nil, fmt.Errorf("weights: GPUWireF32 packed format %s missing blob", s.Format)
		}
		out, err := quant.Unpack(s.Packed)
		if err != nil {
			return nil, err
		}
		if len(out) < n {
			return nil, fmt.Errorf("weights: unpack short for %s", s.Format)
		}
		s.gpuF32 = out[:n]
		return s.gpuF32, nil
	}
	out := make([]float32, n)
	row := make([]float32, s.Cols)
	for r := 0; r < s.Rows; r++ {
		if err := DecodeRow(s, r, row); err != nil {
			return nil, err
		}
		copy(out[r*s.Cols:], row)
	}
	s.gpuF32 = out
	return s.gpuF32, nil
}

// WireRow copies one weight row from the compute-wire view into dst.
func WireRow(s *Store, row int, dst []float32) error {
	if s == nil {
		return fmt.Errorf("weights: nil")
	}
	if row < 0 || row >= s.Rows {
		return fmt.Errorf("weights: row %d out of range", row)
	}
	if len(dst) < s.Cols {
		return fmt.Errorf("weights: dst short")
	}
	w, err := s.GPUWireF32()
	if err != nil {
		return err
	}
	copy(dst[:s.Cols], w[row*s.Cols:(row+1)*s.Cols])
	return nil
}

// GPUFloat32 is an alias for GPUWireF32.
func (s *Store) GPUFloat32() ([]float32, error) { return s.GPUWireF32() }

// MasterF32 returns the pack-source buffer when FormatNone+Float32 (for SIMD DotTile).
func (s *Store) MasterF32() ([]float32, bool) {
	if s == nil || s.Format != quant.FormatNone || s.DType != core.DTypeFloat32 {
		return nil, false
	}
	n := s.Rows * s.Cols
	if len(s.masterF32) < n {
		return nil, false
	}
	return s.masterF32[:n], true
}
