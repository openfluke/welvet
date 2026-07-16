package webgpu

import (
	"fmt"
	"math"

	"github.com/openfluke/webgpu/wgpu"
	"github.com/openfluke/welvet/tiling"
)

type iqParams struct {
	Batch      uint32
	InputSize  uint32
	OutputSize uint32
	Bits       uint32
	ScaleGroup uint32
	Nonlinear  uint32
	MidBits    uint32
	Pad        uint32
}

// DenseGEMVIQ — on-device IQ family GEMV. rawBits = Raw packed as u32 LE words.
func DenseGEMVIQ(scales []float32, rawBits []uint32, bits, scaleGroup int, nonlinear bool, mid float32, x, y []float32, batch, in, out int) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu DenseGEMVIQ: %w", initErr)
	}
	if err := sess.ensureIQPipe(); err != nil {
		return err
	}
	return sess.gemvIQ(scales, rawBits, bits, scaleGroup, nonlinear, mid, x, y, batch, in, out)
}

// DenseGEMVTQ4 — packed Q4_0 transpose GEMV (dX).
func DenseGEMVTQ4(scales []float32, packed []uint32, gy, gx []float32, batch, in, out int) error {
	ensure()
	if !haveGPU || sess == nil {
		return fmt.Errorf("webgpu DenseGEMVTQ4: no device")
	}
	if err := sess.ensureGemvtPipes(); err != nil {
		return err
	}
	return sess.gemvtPacked(sess.pipeQ4T, scales, packed, gy, gx, batch, in, out)
}

// DenseGEMVTQ8 — packed Q8_0 transpose GEMV.
func DenseGEMVTQ8(scales []float32, packed []uint32, gy, gx []float32, batch, in, out int) error {
	ensure()
	if !haveGPU || sess == nil {
		return fmt.Errorf("webgpu DenseGEMVTQ8: no device")
	}
	if err := sess.ensureGemvtPipes(); err != nil {
		return err
	}
	return sess.gemvtPacked(sess.pipeQ8T, scales, packed, gy, gx, batch, in, out)
}

// DenseGEMVTTernary — packed ternary transpose GEMV.
func DenseGEMVTTernary(scales []float32, words []uint32, gy, gx []float32, batch, in, out int) error {
	ensure()
	if !haveGPU || sess == nil {
		return fmt.Errorf("webgpu DenseGEMVTTernary: no device")
	}
	if err := sess.ensureGemvtPipes(); err != nil {
		return err
	}
	return sess.gemvtPacked(sess.pipeTernaryT, scales, words, gy, gx, batch, in, out)
}

// DenseGEMVTBinary — packed binary transpose GEMV.
func DenseGEMVTBinary(scales []float32, words []uint32, gy, gx []float32, batch, in, out int) error {
	ensure()
	if !haveGPU || sess == nil {
		return fmt.Errorf("webgpu DenseGEMVTBinary: no device")
	}
	if err := sess.ensureGemvtPipes(); err != nil {
		return err
	}
	return sess.gemvtPacked(sess.pipeBinaryT, scales, words, gy, gx, batch, in, out)
}

func (s *session) ensureIQPipe() error {
	if s.pipeIQ != nil {
		return nil
	}
	p, err := makePipeline(s.device, ShaderDenseIQ, "welvet-dense-iq")
	if err != nil {
		return err
	}
	s.pipeIQ = p
	return nil
}

func (s *session) ensureGemvtPipes() error {
	if s.pipeQ4T != nil {
		return nil
	}
	var err error
	if s.pipeQ4T, err = makePipeline(s.device, ShaderDenseQ4T, "welvet-q4t"); err != nil {
		return err
	}
	if s.pipeQ8T, err = makePipeline(s.device, ShaderDenseQ8T, "welvet-q8t"); err != nil {
		return err
	}
	if s.pipeTernaryT, err = makePipeline(s.device, ShaderDenseTernaryT, "welvet-tert"); err != nil {
		return err
	}
	if s.pipeBinaryT, err = makePipeline(s.device, ShaderDenseBinaryT, "welvet-bint"); err != nil {
		return err
	}
	return nil
}

func (s *session) gemvIQ(scales []float32, rawBits []uint32, bits, scaleGroup int, nonlinear bool, mid float32, x, y []float32, batch, in, out int) error {
	dev, q := s.device, s.queue
	const wg = 64

	scBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "iq-sc", Contents: wgpu.ToBytes(scales),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer scBuf.Destroy()
	rawBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "iq-raw", Contents: wgpu.ToBytes(rawBits),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer rawBuf.Destroy()
	xBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "iq-x", Contents: wgpu.ToBytes(x[:batch*in]),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer xBuf.Destroy()
	yBytes := uint64(batch * out * 4)
	if yBytes < 64 {
		yBytes = 64
	}
	yBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "iq-y", Size: yBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer yBuf.Destroy()

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
		Label: "iq-p", Contents: wgpu.ToBytes([]iqParams{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()

	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.pipeIQ.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: xBuf, Offset: 0, Size: xBuf.GetSize()},
			{Binding: 2, Buffer: scBuf, Offset: 0, Size: scBuf.GetSize()},
			{Binding: 3, Buffer: rawBuf, Offset: 0, Size: rawBuf.GetSize()},
			{Binding: 4, Buffer: yBuf, Offset: 0, Size: yBuf.GetSize()},
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
	pass.SetPipeline(s.pipeIQ)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(tiling.GPUWorkgroupsX(out, wg), uint32(batch), 1)
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

func (s *session) gemvtPacked(pipe *wgpu.ComputePipeline, scales []float32, packed []uint32, gy, gx []float32, batch, in, out int) error {
	dev, q := s.device, s.queue
	const wg = 64

	scBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "t-sc", Contents: wgpu.ToBytes(scales),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer scBuf.Destroy()
	pkBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "t-pk", Contents: wgpu.ToBytes(packed),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pkBuf.Destroy()
	gyBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "t-gy", Contents: wgpu.ToBytes(gy[:batch*out]),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer gyBuf.Destroy()
	gxBytes := uint64(batch * in * 4)
	if gxBytes < 64 {
		gxBytes = 64
	}
	gxBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "t-gx", Size: gxBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer gxBuf.Destroy()

	p := gpuParams{Batch: uint32(batch), InputSize: uint32(in), OutputSize: uint32(out)}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "t-p", Contents: wgpu.ToBytes([]gpuParams{p}),
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
			{Binding: 1, Buffer: gyBuf, Offset: 0, Size: gyBuf.GetSize()},
			{Binding: 2, Buffer: scBuf, Offset: 0, Size: scBuf.GetSize()},
			{Binding: 3, Buffer: pkBuf, Offset: 0, Size: pkBuf.GetSize()},
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
	pass.SetPipeline(pipe)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(tiling.GPUWorkgroupsX(in, wg), uint32(batch), 1)
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

// ShaderDenseIQ — on-device IQ dequant GEMV (reads tightly packed n-bit codes).
const ShaderDenseIQ = `
struct Params {
    batch: u32, inputSize: u32, outputSize: u32, bits: u32,
    scaleGroup: u32, nonlinear: u32, midBits: u32, _pad: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> raw: array<u32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;

const IQ4NL: array<f32, 16> = array<f32, 16>(
    -1.0, -0.6961928, -0.52507305, -0.3949175,
    -0.28444138, -0.18477343, -0.091050036, 0.0,
    0.0795803, 0.1609302, 0.2461123, 0.33791524,
    0.44070983, 0.562617, 0.72295684, 1.0
);

fn read_bits(bitPos: u32, nBits: u32) -> u32 {
    var v: u32 = 0u;
    for (var i: u32 = 0u; i < nBits; i++) {
        let p = bitPos + i;
        let word = raw[p / 32u];
        let bit = (word >> (p % 32u)) & 1u;
        v |= bit << i;
    }
    return v;
}

fn dequant(q: u32, scale: f32) -> f32 {
    if (params.nonlinear != 0u) {
        return scale * IQ4NL[q & 15u];
    }
    if (params.bits == 1u) {
        if ((q & 1u) != 0u) { return scale; }
        return -scale;
    }
    let mid = bitcast<f32>(params.midBits);
    return scale * (f32(q) - mid);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    let b = gid.y;
    if (o >= params.outputSize || b >= params.batch) { return; }
    var sum: f32 = 0.0;
    let xBase = b * params.inputSize;
    let wBase = o * params.inputSize;
    for (var c: u32 = 0u; c < params.inputSize; c++) {
        let flat = wBase + c;
        let sIdx = flat / params.scaleGroup;
        let j = flat % params.scaleGroup;
        let bitPos = sIdx * params.scaleGroup * params.bits + j * params.bits;
        let q = read_bits(bitPos, params.bits);
        let w = dequant(q, scales[sIdx]);
        sum += X[xBase + c] * w;
    }
    Y[b * params.outputSize + o] = sum;
}
`

// Transpose shaders: one thread per input index; loop outputs.
const ShaderDenseQ4T = `
struct Params { batch: u32, inputSize: u32, outputSize: u32, _pad: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> GY: array<f32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<u32>;
@group(0) @binding(4) var<storage, read_write> GX: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let b = gid.y;
    if (i >= params.inputSize || b >= params.batch) { return; }
    var sum: f32 = 0.0;
    let gyBase = b * params.outputSize;
    for (var o: u32 = 0u; o < params.outputSize; o++) {
        let flat = o * params.inputSize + i;
        let blockIdx = flat / 32u;
        let scale = scales[blockIdx];
        let wordIdx = flat / 8u;
        let nibble = flat % 8u;
        let packed = weights[wordIdx];
        var q = i32((packed >> (nibble * 4u)) & 0xFu);
        if (q > 7) { q -= 16; }
        sum += GY[gyBase + o] * f32(q) * scale;
    }
    GX[b * params.inputSize + i] = sum;
}
`

const ShaderDenseQ8T = `
struct Params { batch: u32, inputSize: u32, outputSize: u32, _pad: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> GY: array<f32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<u32>;
@group(0) @binding(4) var<storage, read_write> GX: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let b = gid.y;
    if (i >= params.inputSize || b >= params.batch) { return; }
    var sum: f32 = 0.0;
    let gyBase = b * params.outputSize;
    for (var o: u32 = 0u; o < params.outputSize; o++) {
        let flat = o * params.inputSize + i;
        let blockIdx = flat / 32u;
        let scale = scales[blockIdx];
        let wordIdx = blockIdx * 8u + (flat % 32u) / 4u;
        let lane = (flat % 32u) % 4u;
        let packed = weights[wordIdx];
        var q = i32((packed >> (lane * 8u)) & 0xFFu);
        if (q > 127) { q -= 256; }
        sum += GY[gyBase + o] * f32(q) * scale;
    }
    GX[b * params.inputSize + i] = sum;
}
`

const ShaderDenseTernaryT = `
struct Params { batch: u32, inputSize: u32, outputSize: u32, _pad: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> GY: array<f32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<u32>;
@group(0) @binding(4) var<storage, read_write> GX: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let b = gid.y;
    if (i >= params.inputSize || b >= params.batch) { return; }
    var sum: f32 = 0.0;
    let gyBase = b * params.outputSize;
    for (var o: u32 = 0u; o < params.outputSize; o++) {
        let flat = o * params.inputSize + i;
        let g = flat / 16u;
        let j = flat % 16u;
        let word = weights[g];
        let scale = scales[g];
        let code = (word >> (j * 2u)) & 3u;
        var tw: f32 = 0.0;
        if (code == 0u) { tw = -scale; }
        else if (code == 2u) { tw = scale; }
        sum += GY[gyBase + o] * tw;
    }
    GX[b * params.inputSize + i] = sum;
}
`

const ShaderDenseBinaryT = `
struct Params { batch: u32, inputSize: u32, outputSize: u32, _pad: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> GY: array<f32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<u32>;
@group(0) @binding(4) var<storage, read_write> GX: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let b = gid.y;
    if (i >= params.inputSize || b >= params.batch) { return; }
    var sum: f32 = 0.0;
    let gyBase = b * params.outputSize;
    for (var o: u32 = 0u; o < params.outputSize; o++) {
        let flat = o * params.inputSize + i;
        let g = flat / 32u;
        let j = flat % 32u;
        let word = weights[g];
        let scale = scales[g];
        var tw: f32 = -scale;
        if (((word >> j) & 1u) != 0u) { tw = scale; }
        sum += GY[gyBase + o] * tw;
    }
    GX[b * params.inputSize + i] = sum;
}
`
