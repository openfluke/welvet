package dense

import (
	"encoding/binary"
	"fmt"
	"math"

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

	case l.Weights.Format == quant.FormatNone && l.Weights.DType == core.DTypeFloat32:
		w, err := l.Weights.GPUWireF32()
		if err != nil {
			return err
		}
		return webgpu.DenseGEMV(w, xF, yF, batch, in, out)

	default:
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

	for b := 0; b < batch; b++ {
		xRow := input.Data[b*in : (b+1)*in]
		gy := dPreF[b*out : (b+1)*out]
		for o := 0; o < out; o++ {
			g := float64(gy[o])
			dw := gradW.Data[o*in : (o+1)*in]
			for i := 0; i < in; i++ {
				dw[i] = core.FromFloat64[T](core.AsFloat64(dw[i]) + g*core.AsFloat64(xRow[i]))
			}
		}
	}
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

	case l.Weights.Format != quant.FormatNone && l.Weights.Packed != nil:
		// Q4_1/Q5: host MatVecT (no f32 W upload to GPU).
		for bat := 0; bat < batch; bat++ {
			gy := dPreF[bat*out : (bat+1)*out]
			gx := make([]float32, in)
			if err := quant.MatVecT(l.Weights.Packed, gy, gx); err != nil {
				return err
			}
			copy(gxF[bat*in:(bat+1)*in], gx)
		}
		return nil

	default:
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
	words := (n + 3) / 4
	out := make([]uint32, words)
	padded := make([]byte, words*4)
	copy(padded, s.Native[:n])
	for i := 0; i < words; i++ {
		out[i] = binary.LittleEndian.Uint32(padded[i*4:])
	}
	return out, scale, true
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
