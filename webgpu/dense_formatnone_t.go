package webgpu

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
	"github.com/openfluke/welvet/tiling"
)

// DenseGEMVTI8 — Int8 transpose GEMV on device.
func DenseGEMVTI8(body []uint32, scale float32, gy, gx []float32, batch, in, out int) error {
	ensure()
	if !haveGPU || sess == nil {
		return fmt.Errorf("webgpu DenseGEMVTI8: no device")
	}
	if err := sess.ensureFormatNoneTPipes(); err != nil {
		return err
	}
	return sess.gemvtI8(body, scale, gy, gx, batch, in, out)
}

// DenseGEMVTU8 — Uint8 affine transpose GEMV on device.
func DenseGEMVTU8(body []uint32, minV, scale float32, gy, gx []float32, batch, in, out int) error {
	ensure()
	if !haveGPU || sess == nil {
		return fmt.Errorf("webgpu DenseGEMVTU8: no device")
	}
	if err := sess.ensureFormatNoneTPipes(); err != nil {
		return err
	}
	return sess.gemvtU8(body, minV, scale, gy, gx, batch, in, out)
}

// DenseGEMVTNative — FormatNone f16/bf16/fp8/fp4 transpose GEMV.
func DenseGEMVTNative(raw []uint32, kind int, gy, gx []float32, batch, in, out int) error {
	ensure()
	if !haveGPU || sess == nil {
		return fmt.Errorf("webgpu DenseGEMVTNative: no device")
	}
	if err := sess.ensureFormatNoneTPipes(); err != nil {
		return err
	}
	return sess.gemvtNative(raw, kind, gy, gx, batch, in, out)
}

// DenseGEMVTExt — FormatNone Ext transpose GEMV.
func DenseGEMVTExt(raw []uint32, kind, bits int, minV, scale float32, gy, gx []float32, batch, in, out int) error {
	ensure()
	if !haveGPU || sess == nil {
		return fmt.Errorf("webgpu DenseGEMVTExt: no device")
	}
	if err := sess.ensureFormatNoneTPipes(); err != nil {
		return err
	}
	return sess.gemvtExt(raw, kind, bits, minV, scale, gy, gx, batch, in, out)
}

func (s *session) ensureFormatNoneTPipes() error {
	if s.pipeFP32T != nil {
		return nil
	}
	var err error
	if s.pipeFP32T, err = makePipeline(s.device, ShaderDenseFP32T, "welvet-fp32t"); err != nil {
		return err
	}
	if s.pipeI8T, err = makePipeline(s.device, ShaderDenseI8T, "welvet-i8t"); err != nil {
		return err
	}
	if s.pipeU8T, err = makePipeline(s.device, ShaderDenseU8T, "welvet-u8t"); err != nil {
		return err
	}
	if s.pipeNativeT, err = makePipeline(s.device, ShaderDenseNativeT, "welvet-nativet"); err != nil {
		return err
	}
	if s.pipeExtT, err = makePipeline(s.device, ShaderDenseExtT, "welvet-extt"); err != nil {
		return err
	}
	return nil
}

func (s *session) gemvtFP32(w, gy, gx []float32, batch, in, out int) error {
	dev, q := s.device, s.queue
	wBuf, err := bufF32(dev, "fp32t-w", w[:out*in])
	if err != nil {
		return err
	}
	defer wBuf.Destroy()
	gyBuf, err := bufF32(dev, "fp32t-gy", gy[:batch*out])
	if err != nil {
		return err
	}
	defer gyBuf.Destroy()
	gxBytes := uint64(batch * in * 4)
	if gxBytes < 64 {
		gxBytes = 64
	}
	gxBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "fp32t-gx", Size: gxBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer gxBuf.Destroy()
	p := gpuParams{Batch: uint32(batch), InputSize: uint32(in), OutputSize: uint32(out)}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "fp32t-p", Contents: wgpu.ToBytes([]gpuParams{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()
	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.pipeFP32T.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: gyBuf, Offset: 0, Size: gyBuf.GetSize()},
			{Binding: 2, Buffer: wBuf, Offset: 0, Size: wBuf.GetSize()},
			{Binding: 3, Buffer: gxBuf, Offset: 0, Size: gxBuf.GetSize()},
		},
	})
	if err != nil {
		return err
	}
	defer bg.Release()
	return dispatchReadGX(dev, q, s.pipeFP32T, bg, gx, gxBuf, batch, in)
}

func (s *session) gemvtI8(body []uint32, scale float32, gy, gx []float32, batch, in, out int) error {
	dev, q := s.device, s.queue
	wBuf, err := bufU32(dev, "i8t-w", body)
	if err != nil {
		return err
	}
	defer wBuf.Destroy()
	gyBuf, err := bufF32(dev, "i8t-gy", gy[:batch*out])
	if err != nil {
		return err
	}
	defer gyBuf.Destroy()
	gxBytes := uint64(batch * in * 4)
	if gxBytes < 64 {
		gxBytes = 64
	}
	gxBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "i8t-gx", Size: gxBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer gxBuf.Destroy()
	p := gpuParamsI8{Batch: uint32(batch), InputSize: uint32(in), OutputSize: uint32(out), Scale: scale}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "i8t-p", Contents: wgpu.ToBytes([]gpuParamsI8{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()
	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.pipeI8T.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: gyBuf, Offset: 0, Size: gyBuf.GetSize()},
			{Binding: 2, Buffer: wBuf, Offset: 0, Size: wBuf.GetSize()},
			{Binding: 3, Buffer: gxBuf, Offset: 0, Size: gxBuf.GetSize()},
		},
	})
	if err != nil {
		return err
	}
	defer bg.Release()
	return dispatchReadGX(dev, q, s.pipeI8T, bg, gx, gxBuf, batch, in)
}

func (s *session) gemvtU8(body []uint32, minV, scale float32, gy, gx []float32, batch, in, out int) error {
	dev, q := s.device, s.queue
	wBuf, err := bufU32(dev, "u8t-w", body)
	if err != nil {
		return err
	}
	defer wBuf.Destroy()
	gyBuf, err := bufF32(dev, "u8t-gy", gy[:batch*out])
	if err != nil {
		return err
	}
	defer gyBuf.Destroy()
	gxBytes := uint64(batch * in * 4)
	if gxBytes < 64 {
		gxBytes = 64
	}
	gxBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "u8t-gx", Size: gxBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer gxBuf.Destroy()
	p := u8Params{
		Batch: uint32(batch), InputSize: uint32(in), OutputSize: uint32(out),
		Min: minV, Scale: scale,
	}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "u8t-p", Contents: wgpu.ToBytes([]u8Params{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()
	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.pipeU8T.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: gyBuf, Offset: 0, Size: gyBuf.GetSize()},
			{Binding: 2, Buffer: wBuf, Offset: 0, Size: wBuf.GetSize()},
			{Binding: 3, Buffer: gxBuf, Offset: 0, Size: gxBuf.GetSize()},
		},
	})
	if err != nil {
		return err
	}
	defer bg.Release()
	return dispatchReadGX(dev, q, s.pipeU8T, bg, gx, gxBuf, batch, in)
}

func (s *session) gemvtNative(raw []uint32, kind int, gy, gx []float32, batch, in, out int) error {
	dev, q := s.device, s.queue
	wBuf, err := bufU32(dev, "natt-w", raw)
	if err != nil {
		return err
	}
	defer wBuf.Destroy()
	gyBuf, err := bufF32(dev, "natt-gy", gy[:batch*out])
	if err != nil {
		return err
	}
	defer gyBuf.Destroy()
	gxBytes := uint64(batch * in * 4)
	if gxBytes < 64 {
		gxBytes = 64
	}
	gxBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "natt-gx", Size: gxBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer gxBuf.Destroy()
	p := f16Params{Batch: uint32(batch), InputSize: uint32(in), OutputSize: uint32(out), Kind: uint32(kind)}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "natt-p", Contents: wgpu.ToBytes([]f16Params{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()
	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.pipeNativeT.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: gyBuf, Offset: 0, Size: gyBuf.GetSize()},
			{Binding: 2, Buffer: wBuf, Offset: 0, Size: wBuf.GetSize()},
			{Binding: 3, Buffer: gxBuf, Offset: 0, Size: gxBuf.GetSize()},
		},
	})
	if err != nil {
		return err
	}
	defer bg.Release()
	return dispatchReadGX(dev, q, s.pipeNativeT, bg, gx, gxBuf, batch, in)
}

func (s *session) gemvtExt(raw []uint32, kind, bits int, minV, scale float32, gy, gx []float32, batch, in, out int) error {
	dev, q := s.device, s.queue
	wBuf, err := bufU32(dev, "extt-w", raw)
	if err != nil {
		return err
	}
	defer wBuf.Destroy()
	gyBuf, err := bufF32(dev, "extt-gy", gy[:batch*out])
	if err != nil {
		return err
	}
	defer gyBuf.Destroy()
	gxBytes := uint64(batch * in * 4)
	if gxBytes < 64 {
		gxBytes = 64
	}
	gxBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "extt-gx", Size: gxBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer gxBuf.Destroy()
	p := extParams{
		Batch: uint32(batch), InputSize: uint32(in), OutputSize: uint32(out),
		Kind: uint32(kind), Bits: uint32(bits), MinV: minV, Scale: scale,
	}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "extt-p", Contents: wgpu.ToBytes([]extParams{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()
	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.pipeExtT.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: gyBuf, Offset: 0, Size: gyBuf.GetSize()},
			{Binding: 2, Buffer: wBuf, Offset: 0, Size: wBuf.GetSize()},
			{Binding: 3, Buffer: gxBuf, Offset: 0, Size: gxBuf.GetSize()},
		},
	})
	if err != nil {
		return err
	}
	defer bg.Release()
	return dispatchReadGX(dev, q, s.pipeExtT, bg, gx, gxBuf, batch, in)
}

func dispatchReadGX(dev *wgpu.Device, q *wgpu.Queue, pipe *wgpu.ComputePipeline, bg *wgpu.BindGroup, gx []float32, gxBuf *wgpu.Buffer, batch, in int) error {
	enc, err := dev.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipe)
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
