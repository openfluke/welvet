package webgpu

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
	"github.com/openfluke/welvet/tiling"
)

type u8Params struct {
	Batch      uint32
	InputSize  uint32
	OutputSize uint32
	Pad        uint32
	Min        float32
	Scale      float32
	_          [2]float32
}

type f16Params struct {
	Batch      uint32
	InputSize  uint32
	OutputSize uint32
	Kind       uint32 // 0=f16 1=bf16 2=fp8e4m3 3=fp8e5m2 4=fp4
}

// DenseGEMVU8 — affine uint8 weights: w = min + q*scale.
func DenseGEMVU8(body []uint32, minV, scale float32, x, y []float32, batch, in, out int) error {
	ensure()
	if !haveGPU || sess == nil {
		return fmt.Errorf("webgpu DenseGEMVU8: no device")
	}
	if err := sess.ensureFormatNonePipes(); err != nil {
		return err
	}
	return sess.gemvU8(body, minV, scale, x, y, batch, in, out)
}

// DenseGEMVNative — FormatNone low-precision: kind 0=f16 1=bf16 2=e4m3 3=e5m2 4=fp4(nibbles).
// raw is little-endian packed native bytes as u32 words.
func DenseGEMVNative(raw []uint32, kind int, x, y []float32, batch, in, out int) error {
	ensure()
	if !haveGPU || sess == nil {
		return fmt.Errorf("webgpu DenseGEMVNative: no device")
	}
	if err := sess.ensureFormatNonePipes(); err != nil {
		return err
	}
	return sess.gemvNative(raw, kind, x, y, batch, in, out)
}

type extParams struct {
	Batch      uint32
	InputSize  uint32
	OutputSize uint32
	Kind       uint32
	Bits       uint32
	_          [3]uint32
	MinV       float32
	Scale      float32
	_pad       [2]float32
}

// DenseGEMVExt — remaining FormatNone dtypes (Uint4/2, NF4, N-bit, wide ints, f64, complex).
func DenseGEMVExt(raw []uint32, kind, bits int, minV, scale float32, x, y []float32, batch, in, out int) error {
	ensure()
	if !haveGPU || sess == nil {
		return fmt.Errorf("webgpu DenseGEMVExt: no device")
	}
	if err := sess.ensureFormatNonePipes(); err != nil {
		return err
	}
	return sess.gemvExt(raw, kind, bits, minV, scale, x, y, batch, in, out)
}

// DenseGEMVTQ4_1 — Q4_1 transpose GEMV.
func DenseGEMVTQ4_1(scales, mins []float32, packed []uint32, gy, gx []float32, batch, in, out int) error {
	ensure()
	if !haveGPU || sess == nil {
		return fmt.Errorf("webgpu DenseGEMVTQ4_1: no device")
	}
	if err := sess.ensureExtraClassic(); err != nil {
		return err
	}
	if err := sess.ensureQ41T(); err != nil {
		return err
	}
	return sess.gemvtQ41(scales, mins, packed, gy, gx, batch, in, out)
}

// DenseGEMVTQ5 — Q5_0/Q5_1 transpose GEMV.
func DenseGEMVTQ5(raw []uint32, blockBytes int, hasMin bool, gy, gx []float32, batch, in, out int) error {
	ensure()
	if !haveGPU || sess == nil {
		return fmt.Errorf("webgpu DenseGEMVTQ5: no device")
	}
	if err := sess.ensureExtraClassic(); err != nil {
		return err
	}
	if err := sess.ensureQ5T(); err != nil {
		return err
	}
	return sess.gemvtQ5(raw, blockBytes, hasMin, gy, gx, batch, in, out)
}

// DenseDW — dW[o,i] = Σ_b x[b,i] * gy[b,o] on device.
func DenseDW(x, gy, dw []float32, batch, in, out int) error {
	ensure()
	if !haveGPU || sess == nil {
		return fmt.Errorf("webgpu DenseDW: no device")
	}
	if err := sess.ensureFormatNonePipes(); err != nil {
		return err
	}
	return sess.gemvDW(x, gy, dw, batch, in, out)
}

func (s *session) ensureFormatNonePipes() error {
	if s.pipeU8 != nil && s.pipeExt != nil {
		return nil
	}
	var err error
	if s.pipeU8 == nil {
		if s.pipeU8, err = makePipeline(s.device, ShaderDenseU8, "welvet-u8"); err != nil {
			return err
		}
	}
	if s.pipeNative == nil {
		if s.pipeNative, err = makePipeline(s.device, ShaderDenseNative, "welvet-native"); err != nil {
			return err
		}
	}
	if s.pipeExt == nil {
		if s.pipeExt, err = makePipeline(s.device, ShaderDenseExt, "welvet-ext"); err != nil {
			return err
		}
	}
	if s.pipeDW == nil {
		if s.pipeDW, err = makePipeline(s.device, ShaderDenseDW, "welvet-dw"); err != nil {
			return err
		}
	}
	return nil
}

func (s *session) ensureQ41T() error {
	if s.pipeQ41T != nil {
		return nil
	}
	p, err := makePipeline(s.device, ShaderDenseQ4_1T, "welvet-q41t")
	if err != nil {
		return err
	}
	s.pipeQ41T = p
	return nil
}

func (s *session) ensureQ5T() error {
	if s.pipeQ5T != nil {
		return nil
	}
	p, err := makePipeline(s.device, ShaderDenseQ5T, "welvet-q5t")
	if err != nil {
		return err
	}
	s.pipeQ5T = p
	return nil
}

func (s *session) gemvU8(body []uint32, minV, scale float32, x, y []float32, batch, in, out int) error {
	dev, q := s.device, s.queue
	wBuf, err := bufU32(dev, "u8-w", body)
	if err != nil {
		return err
	}
	defer wBuf.Destroy()
	xBuf, err := bufF32(dev, "u8-x", x[:batch*in])
	if err != nil {
		return err
	}
	defer xBuf.Destroy()
	yBytes := uint64(batch * out * 4)
	if yBytes < 64 {
		yBytes = 64
	}
	yBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "u8-y", Size: yBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer yBuf.Destroy()
	p := u8Params{
		Batch: uint32(batch), InputSize: uint32(in), OutputSize: uint32(out),
		Min: minV, Scale: scale,
	}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "u8-p", Contents: wgpu.ToBytes([]u8Params{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()
	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.pipeU8.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: xBuf, Offset: 0, Size: xBuf.GetSize()},
			{Binding: 2, Buffer: wBuf, Offset: 0, Size: wBuf.GetSize()},
			{Binding: 3, Buffer: yBuf, Offset: 0, Size: yBuf.GetSize()},
		},
	})
	if err != nil {
		return err
	}
	defer bg.Release()
	return dispatchReadY(dev, q, s.pipeU8, bg, y, yBuf, batch, out, out)
}

func (s *session) gemvExt(raw []uint32, kind, bits int, minV, scale float32, x, y []float32, batch, in, out int) error {
	dev, q := s.device, s.queue
	wBuf, err := bufU32(dev, "ext-w", raw)
	if err != nil {
		return err
	}
	defer wBuf.Destroy()
	xBuf, err := bufF32(dev, "ext-x", x[:batch*in])
	if err != nil {
		return err
	}
	defer xBuf.Destroy()
	yBytes := uint64(batch * out * 4)
	if yBytes < 64 {
		yBytes = 64
	}
	yBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "ext-y", Size: yBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer yBuf.Destroy()
	p := extParams{
		Batch: uint32(batch), InputSize: uint32(in), OutputSize: uint32(out),
		Kind: uint32(kind), Bits: uint32(bits), MinV: minV, Scale: scale,
	}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "ext-p", Contents: wgpu.ToBytes([]extParams{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()
	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.pipeExt.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: xBuf, Offset: 0, Size: xBuf.GetSize()},
			{Binding: 2, Buffer: wBuf, Offset: 0, Size: wBuf.GetSize()},
			{Binding: 3, Buffer: yBuf, Offset: 0, Size: yBuf.GetSize()},
		},
	})
	if err != nil {
		return err
	}
	defer bg.Release()
	return dispatchReadY(dev, q, s.pipeExt, bg, y, yBuf, batch, out, out)
}

func (s *session) gemvNative(raw []uint32, kind int, x, y []float32, batch, in, out int) error {
	dev, q := s.device, s.queue
	wBuf, err := bufU32(dev, "nat-w", raw)
	if err != nil {
		return err
	}
	defer wBuf.Destroy()
	xBuf, err := bufF32(dev, "nat-x", x[:batch*in])
	if err != nil {
		return err
	}
	defer xBuf.Destroy()
	yBytes := uint64(batch * out * 4)
	if yBytes < 64 {
		yBytes = 64
	}
	yBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "nat-y", Size: yBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer yBuf.Destroy()
	p := f16Params{Batch: uint32(batch), InputSize: uint32(in), OutputSize: uint32(out), Kind: uint32(kind)}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "nat-p", Contents: wgpu.ToBytes([]f16Params{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()
	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.pipeNative.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: xBuf, Offset: 0, Size: xBuf.GetSize()},
			{Binding: 2, Buffer: wBuf, Offset: 0, Size: wBuf.GetSize()},
			{Binding: 3, Buffer: yBuf, Offset: 0, Size: yBuf.GetSize()},
		},
	})
	if err != nil {
		return err
	}
	defer bg.Release()
	return dispatchReadY(dev, q, s.pipeNative, bg, y, yBuf, batch, out, out)
}

func (s *session) gemvDW(x, gy, dw []float32, batch, in, out int) error {
	dev, q := s.device, s.queue
	xBuf, err := bufF32(dev, "dw-x", x[:batch*in])
	if err != nil {
		return err
	}
	defer xBuf.Destroy()
	gyBuf, err := bufF32(dev, "dw-gy", gy[:batch*out])
	if err != nil {
		return err
	}
	defer gyBuf.Destroy()
	dwBytes := uint64(out * in * 4)
	if dwBytes < 64 {
		dwBytes = 64
	}
	dwBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "dw", Size: dwBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer dwBuf.Destroy()
	p := gpuParams{Batch: uint32(batch), InputSize: uint32(in), OutputSize: uint32(out)}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "dw-p", Contents: wgpu.ToBytes([]gpuParams{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()
	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.pipeDW.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: xBuf, Offset: 0, Size: xBuf.GetSize()},
			{Binding: 2, Buffer: gyBuf, Offset: 0, Size: gyBuf.GetSize()},
			{Binding: 3, Buffer: dwBuf, Offset: 0, Size: dwBuf.GetSize()},
		},
	})
	if err != nil {
		return err
	}
	defer bg.Release()
	enc, err := dev.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(s.pipeDW)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(tiling.GPUWorkgroupsX(out, 64), uint32(in), 1)
	pass.End()
	cmd, err := enc.Finish(nil)
	if err != nil {
		return err
	}
	q.Submit(cmd)
	outDW, err := readbackF32(dev, q, dwBuf, out*in)
	if err != nil {
		return err
	}
	copy(dw, outDW)
	return nil
}

func (s *session) gemvtQ41(scales, mins []float32, packed []uint32, gy, gx []float32, batch, in, out int) error {
	dev, q := s.device, s.queue
	blocks := (out*in + 31) / 32
	scBuf, err := bufF32(dev, "q41t-sc", scales[:blocks])
	if err != nil {
		return err
	}
	defer scBuf.Destroy()
	mnBuf, err := bufF32(dev, "q41t-mn", mins[:blocks])
	if err != nil {
		return err
	}
	defer mnBuf.Destroy()
	pkBuf, err := bufU32(dev, "q41t-pk", packed[:blocks*4])
	if err != nil {
		return err
	}
	defer pkBuf.Destroy()
	gyBuf, err := bufF32(dev, "q41t-gy", gy[:batch*out])
	if err != nil {
		return err
	}
	defer gyBuf.Destroy()
	gxBytes := uint64(batch * in * 4)
	if gxBytes < 64 {
		gxBytes = 64
	}
	gxBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "q41t-gx", Size: gxBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer gxBuf.Destroy()
	p := q41Params{Batch: uint32(batch), InputSize: uint32(in), OutputSize: uint32(out)}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "q41t-p", Contents: wgpu.ToBytes([]q41Params{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()
	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.pipeQ41T.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: gyBuf, Offset: 0, Size: gyBuf.GetSize()},
			{Binding: 2, Buffer: scBuf, Offset: 0, Size: scBuf.GetSize()},
			{Binding: 3, Buffer: mnBuf, Offset: 0, Size: mnBuf.GetSize()},
			{Binding: 4, Buffer: pkBuf, Offset: 0, Size: pkBuf.GetSize()},
			{Binding: 5, Buffer: gxBuf, Offset: 0, Size: gxBuf.GetSize()},
		},
	})
	if err != nil {
		return err
	}
	defer bg.Release()
	enc, err := dev.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(s.pipeQ41T)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(tiling.GPUWorkgroupsX(in, 64), uint32(batch), 1)
	pass.End()
	cmd, err := enc.Finish(nil)
	if err != nil {
		return err
	}
	q.Submit(cmd)
	outX, err := readbackF32(dev, q, gxBuf, batch*in)
	if err != nil {
		return err
	}
	copy(gx, outX)
	return nil
}

func (s *session) gemvtQ5(raw []uint32, blockBytes int, hasMin bool, gy, gx []float32, batch, in, out int) error {
	dev, q := s.device, s.queue
	rawBuf, err := bufU32(dev, "q5t-raw", raw)
	if err != nil {
		return err
	}
	defer rawBuf.Destroy()
	gyBuf, err := bufF32(dev, "q5t-gy", gy[:batch*out])
	if err != nil {
		return err
	}
	defer gyBuf.Destroy()
	gxBytes := uint64(batch * in * 4)
	if gxBytes < 64 {
		gxBytes = 64
	}
	gxBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "q5t-gx", Size: gxBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer gxBuf.Destroy()
	hm := uint32(0)
	if hasMin {
		hm = 1
	}
	p := q5Params{
		Batch: uint32(batch), InputSize: uint32(in), OutputSize: uint32(out),
		BlockBytes: uint32(blockBytes), HasMin: hm,
	}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "q5t-p", Contents: wgpu.ToBytes([]q5Params{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()
	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.pipeQ5T.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: gyBuf, Offset: 0, Size: gyBuf.GetSize()},
			{Binding: 2, Buffer: rawBuf, Offset: 0, Size: rawBuf.GetSize()},
			{Binding: 3, Buffer: gxBuf, Offset: 0, Size: gxBuf.GetSize()},
		},
	})
	if err != nil {
		return err
	}
	defer bg.Release()
	enc, err := dev.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(s.pipeQ5T)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(tiling.GPUWorkgroupsX(in, 64), uint32(batch), 1)
	pass.End()
	cmd, err := enc.Finish(nil)
	if err != nil {
		return err
	}
	q.Submit(cmd)
	outX, err := readbackF32(dev, q, gxBuf, batch*in)
	if err != nil {
		return err
	}
	copy(gx, outX)
	return nil
}

func dispatchReadY(dev *wgpu.Device, q *wgpu.Queue, pipe *wgpu.ComputePipeline, bg *wgpu.BindGroup, y []float32, yBuf *wgpu.Buffer, batch, out, dispatchOut int) error {
	enc, err := dev.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipe)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(tiling.GPUWorkgroupsX(dispatchOut, 64), uint32(batch), 1)
	pass.End()
	cmd, err := enc.Finish(nil)
	if err != nil {
		return err
	}
	q.Submit(cmd)
	outY, err := readbackF32(dev, q, yBuf, batch*out)
	if err != nil {
		return err
	}
	copy(y, outY)
	return nil
}
