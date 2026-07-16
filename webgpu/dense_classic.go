package webgpu

import (
	"fmt"
	"math"

	"github.com/openfluke/webgpu/wgpu"
	"github.com/openfluke/welvet/tiling"
)

type q41Params struct {
	Batch      uint32
	InputSize  uint32
	OutputSize uint32
	Pad        uint32
}

type kParams struct {
	Batch      uint32
	InputSize  uint32
	OutputSize uint32
	SBBytes    uint32
	Bits       uint32
	HasDmin    uint32
	Mid        uint32
	Pad        uint32
}

type q5Params struct {
	Batch      uint32
	InputSize  uint32
	OutputSize uint32
	BlockBytes uint32
	HasMin     uint32 // 1 = Q5_1
	Pad0       uint32
	Pad1       uint32
	Pad2       uint32
}

// DenseGEMVQ4_1 — on-device Q4_1 (scale+min+nibbles).
func DenseGEMVQ4_1(scales, mins []float32, packed []uint32, x, y []float32, batch, in, out int) error {
	ensure()
	if !haveGPU || sess == nil {
		return fmt.Errorf("webgpu DenseGEMVQ4_1: no device")
	}
	if err := sess.ensureExtraClassic(); err != nil {
		return err
	}
	return sess.gemvQ41(scales, mins, packed, x, y, batch, in, out)
}

// DenseGEMVQ5 — on-device Q5_0/Q5_1 from raw bytes (as u32 words).
func DenseGEMVQ5(raw []uint32, blockBytes int, hasMin bool, x, y []float32, batch, in, out int) error {
	ensure()
	if !haveGPU || sess == nil {
		return fmt.Errorf("webgpu DenseGEMVQ5: no device")
	}
	if err := sess.ensureExtraClassic(); err != nil {
		return err
	}
	return sess.gemvQ5(raw, blockBytes, hasMin, x, y, batch, in, out)
}

// DenseGEMVK — on-device k-quant GEMV from raw superblocks.
func DenseGEMVK(raw []uint32, sbBytes, bits int, hasDmin bool, mid int, x, y []float32, batch, in, out int) error {
	ensure()
	if !haveGPU || sess == nil {
		return fmt.Errorf("webgpu DenseGEMVK: no device")
	}
	if err := sess.ensureExtraClassic(); err != nil {
		return err
	}
	return sess.gemvK(raw, sbBytes, bits, hasDmin, mid, x, y, batch, in, out)
}

// DenseGEMVTIQ — IQ transpose GEMV (dX).
func DenseGEMVTIQ(scales []float32, rawBits []uint32, bits, scaleGroup int, nonlinear bool, mid float32, gy, gx []float32, batch, in, out int) error {
	ensure()
	if !haveGPU || sess == nil {
		return fmt.Errorf("webgpu DenseGEMVTIQ: no device")
	}
	if err := sess.ensureIQPipe(); err != nil {
		return err
	}
	if err := sess.ensureIQTPipe(); err != nil {
		return err
	}
	return sess.gemvtIQ(scales, rawBits, bits, scaleGroup, nonlinear, mid, gy, gx, batch, in, out)
}

// DenseGEMVTK — k-quant transpose GEMV (dX).
func DenseGEMVTK(raw []uint32, sbBytes, bits int, hasDmin bool, mid int, gy, gx []float32, batch, in, out int) error {
	ensure()
	if !haveGPU || sess == nil {
		return fmt.Errorf("webgpu DenseGEMVTK: no device")
	}
	if err := sess.ensureExtraClassic(); err != nil {
		return err
	}
	return sess.gemvtK(raw, sbBytes, bits, hasDmin, mid, gy, gx, batch, in, out)
}

func (s *session) ensureExtraClassic() error {
	if s.pipeQ41 != nil {
		return nil
	}
	var err error
	if s.pipeQ41, err = makePipeline(s.device, ShaderDenseQ4_1, "welvet-q41"); err != nil {
		return err
	}
	if s.pipeQ5, err = makePipeline(s.device, ShaderDenseQ5, "welvet-q5"); err != nil {
		return err
	}
	if s.pipeK, err = makePipeline(s.device, ShaderDenseK, "welvet-k"); err != nil {
		return err
	}
	if s.pipeKT, err = makePipeline(s.device, ShaderDenseKT, "welvet-kt"); err != nil {
		return err
	}
	return nil
}

func (s *session) ensureIQTPipe() error {
	if s.pipeIQT != nil {
		return nil
	}
	p, err := makePipeline(s.device, ShaderDenseIQT, "welvet-iqt")
	if err != nil {
		return err
	}
	s.pipeIQT = p
	return nil
}

func (s *session) gemvQ41(scales, mins []float32, packed []uint32, x, y []float32, batch, in, out int) error {
	dev, q := s.device, s.queue
	blocks := (out*in + 31) / 32
	scBuf, err := bufF32(dev, "q41-sc", scales[:blocks])
	if err != nil {
		return err
	}
	defer scBuf.Destroy()
	mnBuf, err := bufF32(dev, "q41-mn", mins[:blocks])
	if err != nil {
		return err
	}
	defer mnBuf.Destroy()
	pkBuf, err := bufU32(dev, "q41-pk", packed[:blocks*4])
	if err != nil {
		return err
	}
	defer pkBuf.Destroy()
	return s.dispatch5(dev, q, s.pipeQ41, scBuf, mnBuf, pkBuf, x, y, batch, in, out, true)
}

func (s *session) gemvQ5(raw []uint32, blockBytes int, hasMin bool, x, y []float32, batch, in, out int) error {
	dev, q := s.device, s.queue
	rawBuf, err := bufU32(dev, "q5-raw", raw)
	if err != nil {
		return err
	}
	defer rawBuf.Destroy()
	hm := uint32(0)
	if hasMin {
		hm = 1
	}
	p := q5Params{
		Batch: uint32(batch), InputSize: uint32(in), OutputSize: uint32(out),
		BlockBytes: uint32(blockBytes), HasMin: hm,
	}
	return s.dispatchRaw(dev, q, s.pipeQ5, rawBuf, p, x, y, batch, in, out, false)
}

func (s *session) gemvK(raw []uint32, sbBytes, bits int, hasDmin bool, mid int, x, y []float32, batch, in, out int) error {
	dev, q := s.device, s.queue
	rawBuf, err := bufU32(dev, "k-raw", raw)
	if err != nil {
		return err
	}
	defer rawBuf.Destroy()
	hd := uint32(0)
	if hasDmin {
		hd = 1
	}
	p := kParams{
		Batch: uint32(batch), InputSize: uint32(in), OutputSize: uint32(out),
		SBBytes: uint32(sbBytes), Bits: uint32(bits), HasDmin: hd, Mid: uint32(mid),
	}
	return s.dispatchK(dev, q, s.pipeK, rawBuf, p, x, y, batch, in, out, false)
}

func (s *session) gemvtK(raw []uint32, sbBytes, bits int, hasDmin bool, mid int, gy, gx []float32, batch, in, out int) error {
	dev, q := s.device, s.queue
	rawBuf, err := bufU32(dev, "kt-raw", raw)
	if err != nil {
		return err
	}
	defer rawBuf.Destroy()
	hd := uint32(0)
	if hasDmin {
		hd = 1
	}
	p := kParams{
		Batch: uint32(batch), InputSize: uint32(in), OutputSize: uint32(out),
		SBBytes: uint32(sbBytes), Bits: uint32(bits), HasDmin: hd, Mid: uint32(mid),
	}
	return s.dispatchK(dev, q, s.pipeKT, rawBuf, p, gy, gx, batch, out, in, true)
}

func (s *session) gemvtIQ(scales []float32, rawBits []uint32, bits, scaleGroup int, nonlinear bool, mid float32, gy, gx []float32, batch, in, out int) error {
	dev, q := s.device, s.queue
	scBuf, err := bufF32(dev, "iqt-sc", scales)
	if err != nil {
		return err
	}
	defer scBuf.Destroy()
	rawBuf, err := bufU32(dev, "iqt-raw", rawBits)
	if err != nil {
		return err
	}
	defer rawBuf.Destroy()
	gyBuf, err := bufF32(dev, "iqt-gy", gy[:batch*out])
	if err != nil {
		return err
	}
	defer gyBuf.Destroy()
	gxBytes := uint64(batch * in * 4)
	if gxBytes < 64 {
		gxBytes = 64
	}
	gxBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "iqt-gx", Size: gxBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer gxBuf.Destroy()
	nl := uint32(0)
	if nonlinear {
		nl = 1
	}
	p := iqParams{
		Batch: uint32(batch), InputSize: uint32(in), OutputSize: uint32(out),
		Bits: uint32(bits), ScaleGroup: uint32(scaleGroup), Nonlinear: nl,
		MidBits: math.Float32bits(mid),
	}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "iqt-p", Contents: wgpu.ToBytes([]iqParams{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()
	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.pipeIQT.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: gyBuf, Offset: 0, Size: gyBuf.GetSize()},
			{Binding: 2, Buffer: scBuf, Offset: 0, Size: scBuf.GetSize()},
			{Binding: 3, Buffer: rawBuf, Offset: 0, Size: rawBuf.GetSize()},
			{Binding: 4, Buffer: gxBuf, Offset: 0, Size: gxBuf.GetSize()},
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
	pass.SetPipeline(s.pipeIQT)
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

func bufF32(dev *wgpu.Device, label string, data []float32) (*wgpu.Buffer, error) {
	return dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: label, Contents: wgpu.ToBytes(data),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
}

func bufU32(dev *wgpu.Device, label string, data []uint32) (*wgpu.Buffer, error) {
	return dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: label, Contents: wgpu.ToBytes(data),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
}

// dispatch5: bindings 0=params,1=X,2=scales,3=mins,4=packed,5=Y — actually use 0-4 with mins as binding 2 and packed 3, scales...
// Q4_1 layout: 0 params, 1 X, 2 scales, 3 mins, 4 packed, 5 Y — need 6 bindings. Simpler: pack mins into second half of scales buffer.
// We use: 0 params, 1 X, 2 scales, 3 mins, 4 packed+y via dispatchPacked style with 5 bindings by combining mins into scales as interleaved — keep 5: scales, mins as binding 2/3, packed 4? 
// dispatchPacked uses: 0p 1x 2sc 3w 4y. For Q4_1 need mins — use binding 2=scales, 3=mins||packed concatenated?
// Custom 6-binding is fine.

func (s *session) dispatch5(dev *wgpu.Device, q *wgpu.Queue, pipe *wgpu.ComputePipeline,
	sc, mn, pk *wgpu.Buffer, x, y []float32, batch, in, out int, forward bool) error {

	xBuf, err := bufF32(dev, "x", x[:batch*in])
	if err != nil {
		return err
	}
	defer xBuf.Destroy()
	yBytes := uint64(batch * out * 4)
	if yBytes < 64 {
		yBytes = 64
	}
	yBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "y", Size: yBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer yBuf.Destroy()
	p := q41Params{Batch: uint32(batch), InputSize: uint32(in), OutputSize: uint32(out)}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "p", Contents: wgpu.ToBytes([]q41Params{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()
	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: pipe.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: xBuf, Offset: 0, Size: xBuf.GetSize()},
			{Binding: 2, Buffer: sc, Offset: 0, Size: sc.GetSize()},
			{Binding: 3, Buffer: mn, Offset: 0, Size: mn.GetSize()},
			{Binding: 4, Buffer: pk, Offset: 0, Size: pk.GetSize()},
			{Binding: 5, Buffer: yBuf, Offset: 0, Size: yBuf.GetSize()},
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
	pass.SetPipeline(pipe)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(tiling.GPUWorkgroupsX(out, 64), uint32(batch), 1)
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
	_ = forward
	return nil
}

func (s *session) dispatchRaw(dev *wgpu.Device, q *wgpu.Queue, pipe *wgpu.ComputePipeline,
	raw *wgpu.Buffer, p q5Params, x, y []float32, batch, in, out int, _ bool) error {

	xBuf, err := bufF32(dev, "q5-x", x[:batch*in])
	if err != nil {
		return err
	}
	defer xBuf.Destroy()
	yBytes := uint64(batch * out * 4)
	if yBytes < 64 {
		yBytes = 64
	}
	yBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "q5-y", Size: yBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer yBuf.Destroy()
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "q5-p", Contents: wgpu.ToBytes([]q5Params{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()
	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: pipe.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: xBuf, Offset: 0, Size: xBuf.GetSize()},
			{Binding: 2, Buffer: raw, Offset: 0, Size: raw.GetSize()},
			{Binding: 3, Buffer: yBuf, Offset: 0, Size: yBuf.GetSize()},
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
	pass.SetPipeline(pipe)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(tiling.GPUWorkgroupsX(out, 64), uint32(batch), 1)
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

func (s *session) dispatchK(dev *wgpu.Device, q *wgpu.Queue, pipe *wgpu.ComputePipeline,
	raw *wgpu.Buffer, p kParams, xOrGy, yOrGx []float32, batch, dimX, dimY int, transpose bool) error {

	inBuf, err := bufF32(dev, "k-in", xOrGy[:batch*dimX])
	if err != nil {
		return err
	}
	defer inBuf.Destroy()
	outN := batch * dimY
	if transpose {
		outN = batch * dimY // dimY is `in` for transpose output gx
	}
	yBytes := uint64(outN * 4)
	if yBytes < 64 {
		yBytes = 64
	}
	outBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "k-out", Size: yBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer outBuf.Destroy()
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "k-p", Contents: wgpu.ToBytes([]kParams{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()
	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: pipe.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: inBuf, Offset: 0, Size: inBuf.GetSize()},
			{Binding: 2, Buffer: raw, Offset: 0, Size: raw.GetSize()},
			{Binding: 3, Buffer: outBuf, Offset: 0, Size: outBuf.GetSize()},
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
	pass.SetPipeline(pipe)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(tiling.GPUWorkgroupsX(dimY, 64), uint32(batch), 1)
	pass.End()
	_ = transpose
	cmd, err := enc.Finish(nil)
	if err != nil {
		return err
	}
	q.Submit(cmd)
	outY, err := readbackF32(dev, q, outBuf, outN)
	if err != nil {
		return err
	}
	copy(yOrGx, outY)
	return nil
}
