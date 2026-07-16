package dense

import (
	"encoding/binary"
	"fmt"
	"math"
	"strconv"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/webgpu"
	"github.com/openfluke/welvet/weights"
)

// ForwardWebGPU — real device only; prefer on-device packed kernels.
func ForwardWebGPU[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	batch, in, out, err := dims(l, input)
	if err != nil {
		return nil, nil, err
	}
	pre = core.NewTensor[T](batch, out)
	post = core.NewTensor[T](batch, out)
	xF := stageActsF32(input.Data)
	yF := make([]float32, batch*out)

	if err := forwardWebGPUDispatch(l, xF, yF, batch, in, out); err != nil {
		return nil, nil, err
	}
	core.SliceFromFloat32(yF, pre.Data)
	applyBiasAct(pre.Data, post.Data, l.Weights.Bias, l.Core.Activation, batch, out)
	return pre, post, nil
}

func forwardWebGPUDispatch(l *Layer, xF, yF []float32, batch, in, out int) error {
	switch {
	case l.Weights.Format == quant.FormatQ4_0 && l.Weights.Packed != nil:
		scales, packed, ok := q4BlobToSIMD(l.Weights.Packed)
		if !ok {
			return fmt.Errorf("dense: WebGPU Q4_0 projection failed")
		}
		return webgpu.DenseGEMVQ4(scales, packed, xF, yF, batch, in, out)

	case l.Weights.Format == quant.FormatQ4_1 && l.Weights.Packed != nil:
		scales, mins, packed, ok := q41BlobToGPU(l.Weights.Packed)
		if !ok {
			return fmt.Errorf("dense: WebGPU Q4_1 projection failed")
		}
		return webgpu.DenseGEMVQ4_1(scales, mins, packed, xF, yF, batch, in, out)

	case l.Weights.Format == quant.FormatQ5_0 && l.Weights.Packed != nil:
		return webgpu.DenseGEMVQ5(bytesToU32(l.Weights.Packed.Raw), 24, false, xF, yF, batch, in, out)

	case l.Weights.Format == quant.FormatQ5_1 && l.Weights.Packed != nil:
		return webgpu.DenseGEMVQ5(bytesToU32(l.Weights.Packed.Raw), 28, true, xF, yF, batch, in, out)

	case l.Weights.Format == quant.FormatQ8_0 && l.Weights.Packed != nil:
		scales, packed, ok := q8BlobToGPU(l.Weights.Packed)
		if !ok {
			return fmt.Errorf("dense: WebGPU Q8_0 projection failed")
		}
		return webgpu.DenseGEMVQ8(scales, packed, xF, yF, batch, in, out)

	case l.Weights.Format == quant.FormatTernaryPacked && l.Weights.Packed != nil:
		scales, words, ok := ternaryBlobToGPU(l.Weights.Packed)
		if !ok {
			return fmt.Errorf("dense: WebGPU Ternary projection failed")
		}
		return webgpu.DenseGEMVTernary(scales, words, xF, yF, batch, in, out)

	case l.Weights.Format == quant.FormatBinaryPacked && l.Weights.Packed != nil:
		scales, words, ok := binaryBlobToGPU(l.Weights.Packed)
		if !ok {
			return fmt.Errorf("dense: WebGPU Binary projection failed")
		}
		return webgpu.DenseGEMVBinary(scales, words, xF, yF, batch, in, out)

	case isIQ(l.Weights.Format) && l.Weights.Packed != nil:
		return forwardWebGPUIQ(l.Weights.Packed, xF, yF, batch, in, out)

	case l.Weights.Format.IsKQuant() && l.Weights.Packed != nil:
		return forwardWebGPUK(l.Weights.Packed, xF, yF, batch, in, out)

	case l.Weights.Format == quant.FormatNone && l.Weights.DType == core.DTypeInt8:
		u32, scale, ok := nativeInt8AsU32(l.Weights, out*in)
		if !ok {
			return fmt.Errorf("dense: WebGPU Int8 missing native")
		}
		return webgpu.DenseGEMVI8(u32, scale, xF, yF, batch, in, out)

	case l.Weights.Format == quant.FormatNone && l.Weights.DType == core.DTypeUint8:
		body, minV, scale, ok := nativeUint8AsU32(l.Weights, out*in)
		if !ok {
			return fmt.Errorf("dense: WebGPU Uint8 missing native")
		}
		return webgpu.DenseGEMVU8(body, minV, scale, xF, yF, batch, in, out)

	case l.Weights.Format == quant.FormatNone && isNarrowI8(l.Weights.DType):
		u32, scale, ok := narrowAsU32(l.Weights, out*in)
		if !ok {
			return fmt.Errorf("dense: WebGPU narrow %s expand failed", l.Weights.DType)
		}
		return webgpu.DenseGEMVI8(u32, scale, xF, yF, batch, in, out)

	case l.Weights.Format == quant.FormatNone && nativeKind(l.Weights.DType) >= 0:
		raw := bytesToU32(l.Weights.Native)
		return webgpu.DenseGEMVNative(raw, nativeKind(l.Weights.DType), xF, yF, batch, in, out)

	case l.Weights.Format == quant.FormatNone && extKind(l.Weights) >= 0:
		k, bits, minV, scale := extMeta(l.Weights)
		raw := bytesToU32(l.Weights.Native)
		return webgpu.DenseGEMVExt(raw, k, bits, minV, scale, xF, yF, batch, in, out)

	case l.Weights.Format == quant.FormatNone && l.Weights.DType == core.DTypeFloat32:
		w, err := l.Weights.GPUWireF32()
		if err != nil {
			return err
		}
		return webgpu.DenseGEMV(w, xF, yF, batch, in, out)

	default:
		if l.Weights.Format == quant.FormatNone {
			return fmt.Errorf("dense: WebGPU FormatNone unsupported dtype %s", l.Weights.DType)
		}
		w, err := gpuStageF32(l.Weights)
		if err != nil {
			return err
		}
		return webgpu.DenseGEMV(w, xF, yF, batch, in, out)
	}
}

func forwardWebGPUIQ(b *quant.Blob, xF, yF []float32, batch, in, out int) error {
	bits, scaleGroup, nonlinear, mid, ok := iqMeta(b)
	if !ok {
		return fmt.Errorf("dense: IQ meta")
	}
	return webgpu.DenseGEMVIQ(b.Scales, bytesToU32(b.Raw), bits, scaleGroup, nonlinear, mid, xF, yF, batch, in, out)
}

func forwardWebGPUK(b *quant.Blob, xF, yF []float32, batch, in, out int) error {
	spec, ok := quant.KSpecFor(b.Format)
	if !ok {
		return fmt.Errorf("dense: k-quant meta")
	}
	return webgpu.DenseGEMVK(bytesToU32(b.Raw), spec.SBBytes, spec.Bits, spec.HasDmin, spec.Mid, xF, yF, batch, in, out)
}

func iqMeta(b *quant.Blob) (bits, scaleGroup int, nonlinear bool, mid float32, ok bool) {
	if b == nil {
		return 0, 0, false, 0, false
	}
	switch b.Format {
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

func bytesToU32(b []byte) []uint32 {
	n := (len(b) + 3) / 4
	out := make([]uint32, n)
	padded := make([]byte, n*4)
	copy(padded, b)
	for i := 0; i < n; i++ {
		out[i] = binary.LittleEndian.Uint32(padded[i*4:])
	}
	return out
}

// BackwardWebGPU — packed GEMVT on device when available.
func BackwardWebGPU[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	batch, in, out, err := dims(l, input)
	if err != nil {
		return nil, nil, err
	}
	dPreF := make([]float32, batch*out)
	act := l.Core.Activation
	for i := 0; i < batch*out; i++ {
		dPreF[i] = float32(core.AsFloat64(gradOut.Data[i]) * core.AsFloat64(core.ActivateDeriv(pre.Data[i], act)))
	}
	gradIn = core.NewTensor[T](batch, in)
	gradW = core.NewTensor[T](out, in)
	gxF := make([]float32, batch*in)

	if err := backwardWebGPUDispatch(l, dPreF, gxF, batch, in, out); err != nil {
		return nil, nil, err
	}
	core.SliceFromFloat32(gxF, gradIn.Data)

	// dW on device when available.
	xF := stageActsF32(input.Data)
	dwF := make([]float32, out*in)
	if err := webgpu.DenseDW(xF, dPreF, dwF, batch, in, out); err != nil {
		// Fall back to host outer product only if GPU DW unavailable — but policy is no silent fallback.
		return nil, nil, err
	}
	core.SliceFromFloat32(dwF, gradW.Data)
	return gradIn, gradW, nil
}

func backwardWebGPUDispatch(l *Layer, dPreF, gxF []float32, batch, in, out int) error {
	switch {
	case l.Weights.Format == quant.FormatQ4_0 && l.Weights.Packed != nil:
		scales, packed, ok := q4BlobToSIMD(l.Weights.Packed)
		if !ok {
			return fmt.Errorf("dense: WebGPU Q4_0T projection failed")
		}
		return webgpu.DenseGEMVTQ4(scales, packed, dPreF, gxF, batch, in, out)

	case l.Weights.Format == quant.FormatQ8_0 && l.Weights.Packed != nil:
		scales, packed, ok := q8BlobToGPU(l.Weights.Packed)
		if !ok {
			return fmt.Errorf("dense: WebGPU Q8_0T projection failed")
		}
		return webgpu.DenseGEMVTQ8(scales, packed, dPreF, gxF, batch, in, out)

	case l.Weights.Format == quant.FormatTernaryPacked && l.Weights.Packed != nil:
		scales, words, ok := ternaryBlobToGPU(l.Weights.Packed)
		if !ok {
			return fmt.Errorf("dense: WebGPU TernaryT projection failed")
		}
		return webgpu.DenseGEMVTTernary(scales, words, dPreF, gxF, batch, in, out)

	case l.Weights.Format == quant.FormatBinaryPacked && l.Weights.Packed != nil:
		scales, words, ok := binaryBlobToGPU(l.Weights.Packed)
		if !ok {
			return fmt.Errorf("dense: WebGPU BinaryT projection failed")
		}
		return webgpu.DenseGEMVTBinary(scales, words, dPreF, gxF, batch, in, out)

	case l.Weights.Format == quant.FormatQ4_1 && l.Weights.Packed != nil:
		scales, mins, packed, ok := q41BlobToGPU(l.Weights.Packed)
		if !ok {
			return fmt.Errorf("dense: WebGPU Q4_1T projection failed")
		}
		return webgpu.DenseGEMVTQ4_1(scales, mins, packed, dPreF, gxF, batch, in, out)

	case l.Weights.Format == quant.FormatQ5_0 && l.Weights.Packed != nil:
		return webgpu.DenseGEMVTQ5(bytesToU32(l.Weights.Packed.Raw), 24, false, dPreF, gxF, batch, in, out)

	case l.Weights.Format == quant.FormatQ5_1 && l.Weights.Packed != nil:
		return webgpu.DenseGEMVTQ5(bytesToU32(l.Weights.Packed.Raw), 28, true, dPreF, gxF, batch, in, out)

	case isIQ(l.Weights.Format) && l.Weights.Packed != nil:
		bits, scaleGroup, nonlinear, mid, ok := iqMeta(l.Weights.Packed)
		if !ok {
			return fmt.Errorf("dense: IQ T meta")
		}
		return webgpu.DenseGEMVTIQ(l.Weights.Packed.Scales, bytesToU32(l.Weights.Packed.Raw),
			bits, scaleGroup, nonlinear, mid, dPreF, gxF, batch, in, out)

	case l.Weights.Format.IsKQuant() && l.Weights.Packed != nil:
		spec, ok := quant.KSpecFor(l.Weights.Format)
		if !ok {
			return fmt.Errorf("dense: k T meta")
		}
		return webgpu.DenseGEMVTK(bytesToU32(l.Weights.Packed.Raw), spec.SBBytes, spec.Bits, spec.HasDmin, spec.Mid,
			dPreF, gxF, batch, in, out)

	case l.Weights.Format == quant.FormatNone && l.Weights.DType == core.DTypeFloat32:
		w, err := l.Weights.GPUWireF32()
		if err != nil {
			return err
		}
		return webgpu.DenseGEMVT(w, dPreF, gxF, batch, in, out)

	case l.Weights.Format == quant.FormatNone && l.Weights.DType == core.DTypeInt8:
		u32, scale, ok := nativeInt8AsU32(l.Weights, out*in)
		if !ok {
			return fmt.Errorf("dense: WebGPU Int8T missing native")
		}
		return webgpu.DenseGEMVTI8(u32, scale, dPreF, gxF, batch, in, out)

	case l.Weights.Format == quant.FormatNone && l.Weights.DType == core.DTypeUint8:
		body, minV, scale, ok := nativeUint8AsU32(l.Weights, out*in)
		if !ok {
			return fmt.Errorf("dense: WebGPU Uint8T missing native")
		}
		return webgpu.DenseGEMVTU8(body, minV, scale, dPreF, gxF, batch, in, out)

	case l.Weights.Format == quant.FormatNone && isNarrowI8(l.Weights.DType):
		u32, scale, ok := narrowAsU32(l.Weights, out*in)
		if !ok {
			return fmt.Errorf("dense: WebGPU narrow T %s expand failed", l.Weights.DType)
		}
		return webgpu.DenseGEMVTI8(u32, scale, dPreF, gxF, batch, in, out)

	case l.Weights.Format == quant.FormatNone && nativeKind(l.Weights.DType) >= 0:
		raw := bytesToU32(l.Weights.Native)
		return webgpu.DenseGEMVTNative(raw, nativeKind(l.Weights.DType), dPreF, gxF, batch, in, out)

	case l.Weights.Format == quant.FormatNone && extKind(l.Weights) >= 0:
		k, bits, minV, scale := extMeta(l.Weights)
		raw := bytesToU32(l.Weights.Native)
		return webgpu.DenseGEMVTExt(raw, k, bits, minV, scale, dPreF, gxF, batch, in, out)

	default:
		if l.Weights.Format == quant.FormatNone {
			return fmt.Errorf("dense: WebGPU FormatNone T unsupported dtype %s", l.Weights.DType)
		}
		w, err := gpuStageF32(l.Weights)
		if err != nil {
			return err
		}
		return webgpu.DenseGEMVT(w, dPreF, gxF, batch, in, out)
	}
}

func gpuStageF32(s *weights.Store) ([]float32, error) {
	if s == nil {
		return nil, fmt.Errorf("dense: nil weights")
	}
	switch weights.SelectWire(s) {
	case weights.WireF32:
		return s.GPUWireF32()
	default:
		w64, err := s.WireF64()
		if err != nil {
			return nil, err
		}
		out := make([]float32, len(w64))
		for i, v := range w64 {
			out[i] = float32(v)
		}
		return out, nil
	}
}

func stageActsF32[T core.Numeric](x []T) []float32 {
	return core.SliceAsFloat32(x)
}

func nativeInt8AsU32(s *weights.Store, n int) ([]uint32, float32, bool) {
	if s == nil || len(s.Native) < n {
		return nil, 0, false
	}
	scale := s.Scale
	if scale == 0 {
		scale = 1
	}
	return bytesToU32(s.Native[:n]), scale, true
}

func nativeUint8AsU32(s *weights.Store, n int) (body []uint32, minV, scale float32, ok bool) {
	body8, minV, scale, ok := nativeUint8Affine(s, n)
	if !ok {
		return nil, 0, 0, false
	}
	return bytesToU32(body8), minV, scale, true
}

func isNarrowI8(dt core.DType) bool {
	switch dt {
	case core.DTypeInt4, core.DTypeInt2, core.DTypeTernary, core.DTypeBinary:
		return true
	default:
		return false
	}
}

func narrowAsU32(s *weights.Store, n int) ([]uint32, float32, bool) {
	wI8, scale, ok := expandNarrowToI8(s, n)
	if !ok {
		return nil, 0, false
	}
	raw := make([]byte, n)
	for i, v := range wI8 {
		raw[i] = byte(v)
	}
	return bytesToU32(raw), scale, true
}

// nativeKind maps FormatNone low-precision dtypes to ShaderDenseNative kind codes.
// Returns -1 when the dtype is not handled by DenseGEMVNative.
func nativeKind(dt core.DType) int {
	switch dt {
	case core.DTypeFloat16:
		return 0
	case core.DTypeBFloat16:
		return 1
	case core.DTypeFP8E4M3:
		return 2
	case core.DTypeFP8E5M2:
		return 3
	case core.DTypeFP4:
		return 4
	default:
		return -1
	}
}

// extKind returns ShaderDenseExt kind or -1.
func extKind(s *weights.Store) int {
	if s == nil {
		return -1
	}
	k, _, _, _ := extMeta(s)
	return k
}

func extMeta(s *weights.Store) (kind, bits int, minV, scale float32) {
	scale = s.Scale
	if scale == 0 {
		scale = 1
	}
	switch s.DType {
	case core.DTypeUint4:
		if len(s.Native) >= 4 {
			minV = math.Float32frombits(binary.LittleEndian.Uint32(s.Native))
		}
		return 0, 4, minV, scale
	case core.DTypeUint2:
		if len(s.Native) >= 4 {
			minV = math.Float32frombits(binary.LittleEndian.Uint32(s.Native))
		}
		return 1, 2, minV, scale
	case core.DTypeNF4:
		return 2, 4, 0, scale
	case core.DTypeFP6, core.DTypeInt6:
		return 3, 6, 0, scale
	case core.DTypeInt5:
		return 3, 5, 0, scale
	case core.DTypeInt3:
		return 3, 3, 0, scale
	case core.DTypeUint6:
		if len(s.Native) >= 4 {
			minV = math.Float32frombits(binary.LittleEndian.Uint32(s.Native))
		}
		return 4, 6, minV, scale
	case core.DTypeUint5:
		if len(s.Native) >= 4 {
			minV = math.Float32frombits(binary.LittleEndian.Uint32(s.Native))
		}
		return 4, 5, minV, scale
	case core.DTypeUint3:
		if len(s.Native) >= 4 {
			minV = math.Float32frombits(binary.LittleEndian.Uint32(s.Native))
		}
		return 4, 3, minV, scale
	case core.DTypeInt16:
		return 5, 0, 0, scale
	case core.DTypeInt32:
		return 6, 0, 0, scale
	case core.DTypeInt64:
		return 7, 0, 0, scale
	case core.DTypeInt:
		if strconv.IntSize == 64 {
			return 7, 0, 0, scale
		}
		return 6, 0, 0, scale
	case core.DTypeUint16:
		if len(s.Native) >= 4 {
			minV = math.Float32frombits(binary.LittleEndian.Uint32(s.Native))
		}
		return 8, 0, minV, scale
	case core.DTypeUint32:
		if len(s.Native) >= 4 {
			minV = math.Float32frombits(binary.LittleEndian.Uint32(s.Native))
		}
		return 9, 0, minV, scale
	case core.DTypeUint64, core.DTypeUintptr:
		if len(s.Native) >= 4 {
			minV = math.Float32frombits(binary.LittleEndian.Uint32(s.Native))
		}
		return 10, 0, minV, scale
	case core.DTypeUint:
		if len(s.Native) >= 4 {
			minV = math.Float32frombits(binary.LittleEndian.Uint32(s.Native))
		}
		if strconv.IntSize == 64 {
			return 10, 0, minV, scale
		}
		return 9, 0, minV, scale
	case core.DTypeFloat64:
		return 11, 0, 0, 1
	case core.DTypeComplex64:
		return 12, 0, 0, 1
	case core.DTypeComplex128:
		return 13, 0, 0, 1
	default:
		return -1, 0, 0, 0
	}
}

func q8BlobToGPU(b *quant.Blob) (scales []float32, packed []uint32, ok bool) {
	if b == nil || b.Format != quant.FormatQ8_0 {
		return nil, nil, false
	}
	const bw, bb = 32, 36
	n := b.Rows * b.Cols
	blocks := (n + bw - 1) / bw
	if len(b.Raw) < blocks*bb {
		return nil, nil, false
	}
	scales = make([]float32, blocks)
	packed = make([]uint32, blocks*8)
	for bi := 0; bi < blocks; bi++ {
		off := bi * bb
		scales[bi] = math.Float32frombits(uint32(b.Raw[off]) | uint32(b.Raw[off+1])<<8 |
			uint32(b.Raw[off+2])<<16 | uint32(b.Raw[off+3])<<24)
		for k := 0; k < 8; k++ {
			packed[bi*8+k] = binary.LittleEndian.Uint32(b.Raw[off+4+k*4:])
		}
	}
	return scales, packed, true
}

func q41BlobToGPU(b *quant.Blob) (scales, mins []float32, packed []uint32, ok bool) {
	if b == nil || b.Format != quant.FormatQ4_1 {
		return nil, nil, nil, false
	}
	const bw, bb = 32, 24
	n := b.Rows * b.Cols
	blocks := (n + bw - 1) / bw
	if len(b.Raw) < blocks*bb {
		return nil, nil, nil, false
	}
	scales = make([]float32, blocks)
	mins = make([]float32, blocks)
	packed = make([]uint32, blocks*4)
	for bi := 0; bi < blocks; bi++ {
		off := bi * bb
		scales[bi] = quant.GetF32(b.Raw[off:])
		mins[bi] = quant.GetF32(b.Raw[off+4:])
		for k := 0; k < 4; k++ {
			packed[bi*4+k] = binary.LittleEndian.Uint32(b.Raw[off+8+k*4:])
		}
	}
	return scales, mins, packed, true
}

func ternaryBlobToGPU(b *quant.Blob) (scales []float32, words []uint32, ok bool) {
	if b == nil || b.Format != quant.FormatTernaryPacked || len(b.Raw) < 4 {
		return nil, nil, false
	}
	groups := len(b.Raw) / 4
	words = make([]uint32, groups)
	scales = make([]float32, groups)
	for g := 0; g < groups; g++ {
		words[g] = binary.LittleEndian.Uint32(b.Raw[g*4:])
		if g < len(b.Scales) {
			scales[g] = b.Scales[g]
		} else {
			scales[g] = 1
		}
	}
	return scales, words, true
}

func binaryBlobToGPU(b *quant.Blob) (scales []float32, words []uint32, ok bool) {
	if b == nil || b.Format != quant.FormatBinaryPacked || len(b.Raw) < 4 {
		return nil, nil, false
	}
	groups := len(b.Raw) / 4
	words = make([]uint32, groups)
	scales = make([]float32, groups)
	for g := 0; g < groups; g++ {
		words[g] = binary.LittleEndian.Uint32(b.Raw[g*4:])
		if g < len(b.Scales) {
			scales[g] = b.Scales[g]
		} else {
			scales[g] = 1
		}
	}
	return scales, words, true
}
