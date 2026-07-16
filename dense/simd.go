package dense

import (
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/simd"
	"github.com/openfluke/welvet/weights"
)

// ForwardSIMD — Plan 9 where available. Compute wire follows SelectWire:
//
//	WireI8  → DotI8Tile
//	WireU8  → affine uint8 fused (uint8 body × float acts; DotTile on tile scratch)
//	WireF32 → DotTile
//	WireF64 → DotTileF64
//
// Q4_0 / TernaryPacked use fused packed kernels (hard error if projection fails — no wire fallback).
func ForwardSIMD[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if !simd.Enabled() {
		return nil, nil, fmt.Errorf("dense: BackendSIMD requires Plan 9 AVX2/NEON (GOARCH unsupported or kernels not linked)")
	}
	simd.SetInt8DotSimdForward(true)
	batch, in, out, err := dims(l, input)
	if err != nil {
		return nil, nil, err
	}
	pre = core.NewTensor[T](batch, out)
	post = core.NewTensor[T](batch, out)

	switch {
	case l.Weights.Format != quant.FormatNone && l.Weights.Packed != nil:
		if err := forwardSIMDPacked(l, input.Data, pre.Data, batch, in, out); err != nil {
			return nil, nil, err
		}
	default:
		if err := forwardSIMDByWire(l, input.Data, pre.Data, batch, in, out); err != nil {
			return nil, nil, err
		}
	}

	applyBiasAct(pre.Data, post.Data, l.Weights.Bias, l.Core.Activation, batch, out)
	return pre, post, nil
}

func forwardSIMDByWire[T core.Numeric](l *Layer, x, y []T, batch, in, out int) error {
	switch l.Weights.DType {
	case core.DTypeInt4, core.DTypeInt2, core.DTypeTernary, core.DTypeBinary:
		return forwardSIMDNarrowI8(l, x, y, batch, in, out)
	case core.DTypeUint4, core.DTypeUint2, core.DTypeUint16, core.DTypeUint32, core.DTypeUint64,
		core.DTypeUint, core.DTypeUintptr, core.DTypeUint3, core.DTypeUint5, core.DTypeUint6:
		return forwardSIMDUintAffine(l, x, y, batch, in, out)
	case core.DTypeFloat16, core.DTypeBFloat16, core.DTypeFP8E4M3, core.DTypeFP8E5M2,
		core.DTypeFP4:
		return forwardSIMDLowpPacked(l, x, y, batch, in, out)
	case core.DTypeNF4, core.DTypeFP6,
		core.DTypeInt3, core.DTypeInt5, core.DTypeInt6,
		core.DTypeInt16, core.DTypeInt32, core.DTypeInt64, core.DTypeInt,
		core.DTypeComplex64, core.DTypeComplex128, core.DTypeFloat64:
		// Stream DecodeRow(F64) → DotTile(F64) — no WireF64 full-matrix cache.
		return forwardSIMDStreamF64(l, x, y, batch, in, out)
	}
	switch weights.SelectWire(l.Weights) {
	case weights.WireI8:
		return forwardSIMDInt8(l, x, y, batch, in, out)
	case weights.WireU8:
		return forwardSIMDUint8(l, x, y, batch, in, out)
	case weights.WireF32:
		// Float32 Master or DecodeRow stream — never GPUWireF32 cache.
		if l.Weights.DType == core.DTypeFloat32 {
			return forwardSIMDMasterF32(l, x, y, batch, in, out)
		}
		return forwardSIMDStreamF32(l, x, y, batch, in, out)
	default:
		return forwardSIMDStreamF64(l, x, y, batch, in, out)
	}
}

// forwardSIMDUintAffine — DecodeRow stream + DotTile for unsigned affine dtypes.
func forwardSIMDUintAffine[T core.Numeric](l *Layer, x, y []T, batch, in, out int) error {
	wRow := make([]float32, in)
	for b := 0; b < batch; b++ {
		xRow := core.SliceAsFloat32(x[b*in : (b+1)*in])
		yRow := make([]float32, out)
		for o := 0; o < out; o++ {
			if err := weights.DecodeRow(l.Weights, o, wRow); err != nil {
				return err
			}
			yRow[o] = float32(simd.DotTile(xRow, wRow, 0, in, 0))
		}
		core.SliceFromFloat32(yRow, y[b*out:(b+1)*out])
	}
	return nil
}

// forwardSIMDNarrowI8 expands Int4/Int2/Ternary/Binary FormatNone into int8 + DotI8Tile.
func forwardSIMDNarrowI8[T core.Numeric](l *Layer, x, y []T, batch, in, out int) error {
	wI8, scale, ok := expandNarrowToI8(l.Weights, out*in)
	if !ok {
		return fmt.Errorf("dense: SIMD narrow dtype %s expand failed", l.Weights.DType)
	}
	xq := make([]int8, in)
	for b := 0; b < batch; b++ {
		xF := core.SliceAsFloat32(x[b*in : (b+1)*in])
		actScale := quantizeActsI8(xF, xq)
		yF := make([]float32, out)
		for o := 0; o < out; o++ {
			acc := simd.DotI8Tile(xq, wI8, 0, o*in, in, 0)
			yF[o] = float32(acc) * actScale * scale
		}
		core.SliceFromFloat32(yF, y[b*out:(b+1)*out])
	}
	return nil
}

func expandNarrowToI8(s *weights.Store, n int) ([]int8, float32, bool) {
	if s == nil || len(s.Native) == 0 {
		return nil, 0, false
	}
	scale := s.Scale
	if scale == 0 {
		scale = 1
	}
	raw := s.Native
	out := make([]int8, n)
	switch s.DType {
	case core.DTypeInt4:
		for i := 0; i < n; i++ {
			var q int8
			if i%2 == 0 {
				q = int8(raw[i/2] & 0xF)
			} else {
				q = int8(raw[i/2] >> 4)
			}
			if q > 7 {
				q -= 16
			}
			out[i] = q
		}
	case core.DTypeInt2:
		for i := 0; i < n; i++ {
			code := (raw[i/4] >> (uint(i%4) * 2)) & 3
			out[i] = int8(code) - 2 // {0,1,2,3} → {-2,-1,0,1}
		}
	case core.DTypeTernary:
		for i := 0; i < n; i++ {
			code := (raw[i/4] >> (uint(i%4) * 2)) & 3
			out[i] = int8(code) - 1 // {0,1,2} → {-1,0,+1}
		}
	case core.DTypeBinary:
		for i := 0; i < n; i++ {
			if raw[i/8]&(1<<uint(i%8)) != 0 {
				out[i] = 1
			} else {
				out[i] = -1
			}
		}
	default:
		return nil, 0, false
	}
	return out, scale, true
}

func forwardSIMDF32[T core.Numeric](l *Layer, x, y []T, batch, in, out int) error {
	return forwardSIMDStreamF32(l, x, y, batch, in, out)
}

func forwardSIMDMasterF32[T core.Numeric](l *Layer, x, y []T, batch, in, out int) error {
	w, ok := l.Weights.MasterF32()
	if !ok || len(w) < out*in {
		return forwardSIMDStreamF32(l, x, y, batch, in, out)
	}
	for b := 0; b < batch; b++ {
		xRow := core.SliceAsFloat32(x[b*in : (b+1)*in])
		yRow := make([]float32, out)
		for o := 0; o < out; o++ {
			yRow[o] = float32(simd.DotTile(xRow, w[o*in:(o+1)*in], 0, in, 0))
		}
		core.SliceFromFloat32(yRow, y[b*out:(b+1)*out])
	}
	return nil
}

// forwardSIMDStreamF32 — DecodeRow per output row (no full-matrix wire cache) + DotTile.
func forwardSIMDStreamF32[T core.Numeric](l *Layer, x, y []T, batch, in, out int) error {
	wRow := make([]float32, in)
	for b := 0; b < batch; b++ {
		xRow := core.SliceAsFloat32(x[b*in : (b+1)*in])
		yRow := make([]float32, out)
		for o := 0; o < out; o++ {
			if err := weights.DecodeRow(l.Weights, o, wRow); err != nil {
				return err
			}
			yRow[o] = float32(simd.DotTile(xRow, wRow, 0, in, 0))
		}
		core.SliceFromFloat32(yRow, y[b*out:(b+1)*out])
	}
	return nil
}

// forwardSIMDStreamF64 — DecodeRowF64 per row (no WireF64 cache) + DotTileF64.
func forwardSIMDStreamF64[T core.Numeric](l *Layer, x, y []T, batch, in, out int) error {
	wRow := make([]float64, in)
	for b := 0; b < batch; b++ {
		xRow := core.SliceAsFloat64(x[b*in : (b+1)*in])
		yRow := make([]float64, out)
		for o := 0; o < out; o++ {
			if err := weights.DecodeRowF64(l.Weights, o, wRow); err != nil {
				return err
			}
			yRow[o] = simd.DotTileF64(xRow, wRow, 0, in, 0)
		}
		core.SliceFromFloat64(yRow, y[b*out:(b+1)*out])
	}
	return nil
}

// forwardSIMDLowpPacked — Float16/BF16/FP8/FP4 from native bytes; HW DotTile MAC, no Wire cache.
func forwardSIMDLowpPacked[T core.Numeric](l *Layer, x, y []T, batch, in, out int) error {
	raw := l.Weights.Native
	if len(raw) == 0 {
		return fmt.Errorf("dense: SIMD lowp missing native %s", l.Weights.DType)
	}
	dt := l.Weights.DType
	for b := 0; b < batch; b++ {
		xRow := core.SliceAsFloat32(x[b*in : (b+1)*in])
		yRow := make([]float32, out)
		for o := 0; o < out; o++ {
			i0 := o * in
			var sum float64
			switch dt {
			case core.DTypeFloat16:
				sum = simd.DotF16Packed(xRow, raw, i0, in, 0)
			case core.DTypeBFloat16:
				sum = simd.DotBF16Packed(xRow, raw, i0, in, 0)
			case core.DTypeFP8E4M3:
				sum = simd.DotFP8Packed(xRow, raw, i0, in, 0, 0)
			case core.DTypeFP8E5M2:
				sum = simd.DotFP8Packed(xRow, raw, i0, in, 1, 0)
			case core.DTypeFP4:
				sum = simd.DotFP4Packed(xRow, raw, i0, in, 0)
			default:
				return fmt.Errorf("dense: SIMD lowp unexpected %s", dt)
			}
			yRow[o] = float32(sum)
		}
		core.SliceFromFloat32(yRow, y[b*out:(b+1)*out])
	}
	return nil
}

func forwardSIMDF64[T core.Numeric](l *Layer, x, y []T, batch, in, out int) error {
	return forwardSIMDStreamF64(l, x, y, batch, in, out)
}

func forwardSIMDQ4_0[T core.Numeric](l *Layer, x, y []T, batch, in, out int) error {
	scales, packed, ok := q4BlobToSIMD(l.Weights.Packed)
	if !ok {
		return fmt.Errorf("dense: Q4_0 blob projection failed")
	}
	for b := 0; b < batch; b++ {
		xRow := core.SliceAsFloat32(x[b*in : (b+1)*in])
		yF := make([]float32, out)
		o := 0
		for ; o+4 <= out && in%32 == 0; o += 4 {
			var tmp [4]float32
			simd.DotQ4_0Rows4(xRow, scales, packed, o*in, in, tmp[:])
			copy(yF[o:o+4], tmp[:])
		}
		for ; o < out; o++ {
			yF[o] = float32(simd.DotQ4_0Row(xRow, scales, packed, o*in, in, 0))
		}
		core.SliceFromFloat32(yF, y[b*out:(b+1)*out])
	}
	return nil
}

func forwardSIMDInt8[T core.Numeric](l *Layer, x, y []T, batch, in, out int) error {
	wI8, ok := nativeInt8Weights(l.Weights, out*in)
	if !ok {
		return fmt.Errorf("dense: SIMD Int8 missing native weights")
	}
	scale := l.Weights.Scale
	if scale == 0 {
		scale = 1
	}
	xq := make([]int8, in)
	for b := 0; b < batch; b++ {
		xF := core.SliceAsFloat32(x[b*in : (b+1)*in])
		actScale := quantizeActsI8(xF, xq)
		yF := make([]float32, out)
		for o := 0; o < out; o++ {
			acc := simd.DotI8Tile(xq, wI8, 0, o*in, in, 0)
			yF[o] = float32(acc) * actScale * scale
		}
		core.SliceFromFloat32(yF, y[b*out:(b+1)*out])
	}
	return nil
}

// forwardSIMDUint8 — affine uint8 storage (min + q*scale). Fused: no full DecodeRow;
// expand q into a scratch tile and DotTile, then add min*sum(x).
func forwardSIMDUint8[T core.Numeric](l *Layer, x, y []T, batch, in, out int) error {
	body, minV, scale, ok := nativeUint8Affine(l.Weights, out*in)
	if !ok {
		return fmt.Errorf("dense: SIMD Uint8 missing native affine weights")
	}
	scratch := make([]float32, in)
	for b := 0; b < batch; b++ {
		xRow := core.SliceAsFloat32(x[b*in : (b+1)*in])
		var sumX float64
		for _, v := range xRow {
			sumX += float64(v)
		}
		yF := make([]float32, out)
		for o := 0; o < out; o++ {
			off := o * in
			for i := 0; i < in; i++ {
				scratch[i] = float32(body[off+i]) * scale
			}
			yF[o] = float32(simd.DotTile(xRow, scratch, 0, in, 0) + float64(minV)*sumX)
		}
		core.SliceFromFloat32(yF, y[b*out:(b+1)*out])
	}
	return nil
}

func forwardSIMDTernary[T core.Numeric](l *Layer, x, y []T, batch, in, out int) error {
	simd.SetBitNetTernarySimdForward(true)
	b := l.Weights.Packed
	if b == nil || len(b.Raw) < 4 {
		return fmt.Errorf("dense: ternary packed missing")
	}
	const group = 16
	n := out * in
	nGroup := (n + group - 1) / group
	if len(b.Raw) < nGroup*4 {
		return fmt.Errorf("dense: ternary raw short")
	}
	for bat := 0; bat < batch; bat++ {
		xF := core.SliceAsFloat32(x[bat*in : (bat+1)*in])
		xq := make([]int8, ((in+31)/32)*32)
		actScale := quantizeActsI8(xF, xq[:in])
		for i := in; i < len(xq); i++ {
			xq[i] = 0
		}
		yF := make([]float32, out)
		for o := 0; o < out; o++ {
			sum := 0.0
			for c0 := 0; c0 < in; c0 += group {
				gIdx := (o*in + c0) / group
				if gIdx*4+4 > len(b.Raw) {
					break
				}
				word := uint32(b.Raw[gIdx*4]) | uint32(b.Raw[gIdx*4+1])<<8 |
					uint32(b.Raw[gIdx*4+2])<<16 | uint32(b.Raw[gIdx*4+3])<<24
				scale := float32(1)
				if gIdx < len(b.Scales) {
					scale = b.Scales[gIdx]
				}
				codes := make([]uint8, 32)
				for j := range codes {
					codes[j] = 1 // ternary 0
				}
				nCol := group
				if c0+nCol > in {
					nCol = in - c0
				}
				var sumAct int32
				for j := 0; j < nCol; j++ {
					codes[j] = uint8((word >> uint(j*2)) & 3)
					sumAct += int32(xq[c0+j])
				}
				actsWin := make([]int8, 32)
				copy(actsWin, xq[c0:c0+nCol])
				acc := simd.BitNetTernaryCodeRowDot(codes, actsWin, 32)
				sum += float64(acc-sumAct) * float64(scale) * float64(actScale)
			}
			yF[o] = float32(sum)
		}
		core.SliceFromFloat32(yF, y[bat*out:(bat+1)*out])
	}
	return nil
}

// BackwardSIMD — saxpy on the compute wire (f32 / f64 / native i8·scale / affine u8).
func BackwardSIMD[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if !simd.Enabled() {
		return nil, nil, fmt.Errorf("dense: BackendSIMD requires Plan 9 AVX2/NEON")
	}
	batch, in, out, err := dims(l, input)
	if err != nil {
		return nil, nil, err
	}

	dPre := make([]float64, batch*out)
	act := l.Core.Activation
	for i := 0; i < batch*out; i++ {
		dPre[i] = core.AsFloat64(gradOut.Data[i]) * core.AsFloat64(core.ActivateDeriv(pre.Data[i], act))
	}
	gradIn = core.NewTensor[T](batch, in)
	gradW = core.NewTensor[T](out, in)

	if l.Weights.Format != quant.FormatNone && l.Weights.Packed != nil {
		if err := backwardSIMDPacked(l, dPre, input, gradIn, gradW, batch, in, out); err != nil {
			return nil, nil, err
		}
		return gradIn, gradW, nil
	}

	dW64 := make([]float64, out*in)

	wire := weights.SelectWire(l.Weights)
	for b := 0; b < batch; b++ {
		x64 := core.SliceAsFloat64(input.Data[b*in : (b+1)*in])
		x32 := core.SliceAsFloat32(input.Data[b*in : (b+1)*in])
		gy := dPre[b*out : (b+1)*out]
		gx64 := make([]float64, in)
		switch wire {
		case weights.WireF32:
			wRow := make([]float32, in)
			for o := 0; o < out; o++ {
				g := gy[o]
				if err := weights.DecodeRow(l.Weights, o, wRow); err != nil {
					return nil, nil, err
				}
				simd.SaxpyF32AccF64(gx64, g, wRow, in)
				simd.SaxpyF32AccF64(dW64[o*in:(o+1)*in], g, x32, in)
			}
		case weights.WireI8:
			wI8, ok := nativeInt8Weights(l.Weights, out*in)
			if !ok {
				return nil, nil, fmt.Errorf("dense: SIMD Int8 bwd missing native")
			}
			scale := float64(l.Weights.Scale)
			if scale == 0 {
				scale = 1
			}
			wRow := make([]float32, in)
			for o := 0; o < out; o++ {
				g := gy[o]
				off := o * in
				for i := 0; i < in; i++ {
					wRow[i] = float32(float64(wI8[off+i]) * scale)
				}
				simd.SaxpyF32AccF64(gx64, g, wRow, in)
				simd.SaxpyF32AccF64(dW64[o*in:(o+1)*in], g, x32, in)
			}
		case weights.WireU8:
			body, minV, scale, ok := nativeUint8Affine(l.Weights, out*in)
			if !ok {
				return nil, nil, fmt.Errorf("dense: SIMD Uint8 bwd missing native")
			}
			wRow := make([]float32, in)
			for o := 0; o < out; o++ {
				g := gy[o]
				off := o * in
				for i := 0; i < in; i++ {
					wRow[i] = float32(body[off+i])*scale + minV
				}
				simd.SaxpyF32AccF64(gx64, g, wRow, in)
				simd.SaxpyF32AccF64(dW64[o*in:(o+1)*in], g, x32, in)
			}
		default:
			wRow := make([]float64, in)
			for o := 0; o < out; o++ {
				g := gy[o]
				if err := weights.DecodeRowF64(l.Weights, o, wRow); err != nil {
					return nil, nil, err
				}
				simd.SaxpyF64AccF64(gx64, g, wRow, in)
				simd.SaxpyF64AccF64(dW64[o*in:(o+1)*in], g, x64, in)
			}
		}
		core.SliceFromFloat64(gx64, gradIn.Data[b*in:(b+1)*in])
	}
	core.SliceFromFloat64(dW64, gradW.Data)
	return gradIn, gradW, nil
}

func nativeInt8Weights(s *weights.Store, n int) ([]int8, bool) {
	if s == nil || len(s.Native) < n {
		return nil, false
	}
	return unsafe.Slice((*int8)(unsafe.Pointer(&s.Native[0])), n), true
}

func nativeUint8Affine(s *weights.Store, n int) (body []uint8, minV, scale float32, ok bool) {
	if s == nil || len(s.Native) < 4+n {
		return nil, 0, 0, false
	}
	minV = math.Float32frombits(binary.LittleEndian.Uint32(s.Native))
	scale = s.Scale
	if scale == 0 {
		scale = 1
	}
	body = s.Native[4 : 4+n]
	return body, minV, scale, true
}

func quantizeActsI8(x []float32, out []int8) float32 {
	maxAbs := float32(0)
	for _, v := range x {
		a := float32(math.Abs(float64(v)))
		if a > maxAbs {
			maxAbs = a
		}
	}
	scale := maxAbs / 127
	if scale == 0 {
		scale = 1
	}
	for i, v := range x {
		q := int(math.Round(float64(v / scale)))
		if q > 127 {
			q = 127
		}
		if q < -128 {
			q = -128
		}
		out[i] = int8(q)
	}
	return scale
}

func q4BlobToSIMD(b *quant.Blob) (scales []float32, packed []uint32, ok bool) {
	if b == nil || b.Format != quant.FormatQ4_0 || len(b.Raw) < 20 {
		return nil, nil, false
	}
	const bw = 32
	n := b.Rows * b.Cols
	blocks := (n + bw - 1) / bw
	if len(b.Raw) < blocks*20 {
		return nil, nil, false
	}
	scales = make([]float32, blocks)
	packed = make([]uint32, blocks*4)
	for bi := 0; bi < blocks; bi++ {
		off := bi * 20
		scales[bi] = math.Float32frombits(uint32(b.Raw[off]) | uint32(b.Raw[off+1])<<8 | uint32(b.Raw[off+2])<<16 | uint32(b.Raw[off+3])<<24)
		for k := 0; k < 4; k++ {
			packed[bi*4+k] = uint32(b.Raw[off+4+k*4]) |
				uint32(b.Raw[off+5+k*4])<<8 |
				uint32(b.Raw[off+6+k*4])<<16 |
				uint32(b.Raw[off+7+k*4])<<24
		}
	}
	return scales, packed, true
}
