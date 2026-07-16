package webgpu

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
	"github.com/openfluke/welvet/tiling"
)

// DenseGEMVQ8 — on-device Q8_0. scales[blocks], i8 weights as u32 words (8 words / block of 32).
func DenseGEMVQ8(scales []float32, packed []uint32, x, y []float32, batch, in, out int) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu DenseGEMVQ8: %w", initErr)
	}
	blocks := (out*in + 31) / 32
	if len(scales) < blocks || len(packed) < blocks*8 || len(x) < batch*in || len(y) < batch*out {
		return fmt.Errorf("webgpu DenseGEMVQ8: shape")
	}
	return sess.gemvQ8(scales, packed, x, y, batch, in, out)
}

// DenseGEMVTernary — on-device BitNet-style ternary (16 codes / u32, per-group scales).
func DenseGEMVTernary(scales []float32, words []uint32, x, y []float32, batch, in, out int) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu DenseGEMVTernary: %w", initErr)
	}
	groups := (out*in + 15) / 16
	if len(scales) < groups || len(words) < groups || len(x) < batch*in || len(y) < batch*out {
		return fmt.Errorf("webgpu DenseGEMVTernary: shape")
	}
	return sess.gemvTernary(scales, words, x, y, batch, in, out)
}

// DenseGEMVBinary — on-device binary (±scale), 32 bits / u32, per-group scales.
func DenseGEMVBinary(scales []float32, words []uint32, x, y []float32, batch, in, out int) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu DenseGEMVBinary: %w", initErr)
	}
	groups := (out*in + 31) / 32
	if len(scales) < groups || len(words) < groups || len(x) < batch*in || len(y) < batch*out {
		return fmt.Errorf("webgpu DenseGEMVBinary: shape")
	}
	return sess.gemvBinary(scales, words, x, y, batch, in, out)
}

func (s *session) ensureExtraPipes() error {
	if s.pipeQ8 != nil {
		return nil
	}
	var err error
	s.pipeQ8, err = makePipeline(s.device, ShaderDenseQ8, "welvet-dense-q8")
	if err != nil {
		return err
	}
	s.pipeTernary, err = makePipeline(s.device, ShaderDenseTernary, "welvet-dense-ternary")
	if err != nil {
		return err
	}
	s.pipeBinary, err = makePipeline(s.device, ShaderDenseBinary, "welvet-dense-binary")
	return err
}

func (s *session) gemvQ8(scales []float32, packed []uint32, x, y []float32, batch, in, out int) error {
	if err := s.ensureExtraPipes(); err != nil {
		return err
	}
	const wg = 64
	dev, q := s.device, s.queue
	blocks := (out*in + 31) / 32

	scBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-q8-sc", Contents: wgpu.ToBytes(scales[:blocks]),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer scBuf.Destroy()
	pkBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-q8-pk", Contents: wgpu.ToBytes(packed[:blocks*8]),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pkBuf.Destroy()
	return s.dispatchPacked(dev, q, s.pipeQ8, scBuf, pkBuf, x, y, batch, in, out, wg, true)
}

func (s *session) gemvTernary(scales []float32, words []uint32, x, y []float32, batch, in, out int) error {
	if err := s.ensureExtraPipes(); err != nil {
		return err
	}
	groups := (out*in + 15) / 16
	dev, q := s.device, s.queue
	scBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-ter-sc", Contents: wgpu.ToBytes(scales[:groups]),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer scBuf.Destroy()
	wBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-ter-w", Contents: wgpu.ToBytes(words[:groups]),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer wBuf.Destroy()
	return s.dispatchPacked(dev, q, s.pipeTernary, scBuf, wBuf, x, y, batch, in, out, 64, true)
}

func (s *session) gemvBinary(scales []float32, words []uint32, x, y []float32, batch, in, out int) error {
	if err := s.ensureExtraPipes(); err != nil {
		return err
	}
	groups := (out*in + 31) / 32
	dev, q := s.device, s.queue
	scBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-bin-sc", Contents: wgpu.ToBytes(scales[:groups]),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer scBuf.Destroy()
	wBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-bin-w", Contents: wgpu.ToBytes(words[:groups]),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer wBuf.Destroy()
	return s.dispatchPacked(dev, q, s.pipeBinary, scBuf, wBuf, x, y, batch, in, out, 64, true)
}

func (s *session) dispatchPacked(dev *wgpu.Device, q *wgpu.Queue, pipe *wgpu.ComputePipeline,
	scBuf, wBuf *wgpu.Buffer, x, y []float32, batch, in, out, wg int, _ bool) error {

	xBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-pk-x", Contents: wgpu.ToBytes(x[:batch*in]),
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
		Label: "welvet-pk-y", Size: yBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer yBuf.Destroy()

	p := gpuParams{Batch: uint32(batch), InputSize: uint32(in), OutputSize: uint32(out)}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-pk-p", Contents: wgpu.ToBytes([]gpuParams{p}),
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
			{Binding: 2, Buffer: scBuf, Offset: 0, Size: scBuf.GetSize()},
			{Binding: 3, Buffer: wBuf, Offset: 0, Size: wBuf.GetSize()},
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
	pass.SetPipeline(pipe)
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

const ShaderDenseQ8 = `
struct Params { batch: u32, inputSize: u32, outputSize: u32, _pad: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<u32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;

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
        let blockIdx = flat / 32u;
        let j = flat % 32u;
        let scale = scales[blockIdx];
        let packed = weights[blockIdx * 8u + (j / 4u)];
        let lane = j % 4u;
        var q = i32((packed >> (lane * 8u)) & 0xFFu);
        if (q > 127) { q -= 256; }
        sum += X[xBase + c] * f32(q) * scale;
    }
    Y[b * params.outputSize + o] = sum;
}
`

const ShaderDenseTernary = `
struct Params { batch: u32, inputSize: u32, outputSize: u32, _pad: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<u32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    let b = gid.y;
    if (o >= params.outputSize || b >= params.batch) { return; }
    var sum: f32 = 0.0;
    let xBase = b * params.inputSize;
    let wBase = o * params.inputSize;
    for (var c: u32 = 0u; c < params.inputSize; c += 16u) {
        let g = (wBase + c) / 16u;
        let word = weights[g];
        let scale = scales[g];
        var n = 16u;
        if (c + n > params.inputSize) { n = params.inputSize - c; }
        for (var j: u32 = 0u; j < n; j++) {
            let code = (word >> (j * 2u)) & 3u;
            var tw: f32 = 0.0;
            if (code == 0u) { tw = -scale; }
            else if (code == 2u) { tw = scale; }
            sum += X[xBase + c + j] * tw;
        }
    }
    Y[b * params.outputSize + o] = sum;
}
`

const ShaderDenseBinary = `
struct Params { batch: u32, inputSize: u32, outputSize: u32, _pad: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<u32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    let b = gid.y;
    if (o >= params.outputSize || b >= params.batch) { return; }
    var sum: f32 = 0.0;
    let xBase = b * params.inputSize;
    let wBase = o * params.inputSize;
    for (var c: u32 = 0u; c < params.inputSize; c += 32u) {
        let g = (wBase + c) / 32u;
        let word = weights[g];
        let scale = scales[g];
        var n = 32u;
        if (c + n > params.inputSize) { n = params.inputSize - c; }
        for (var j: u32 = 0u; j < n; j++) {
            var tw: f32 = -scale;
            if (((word >> j) & 1u) != 0u) { tw = scale; }
            sum += X[xBase + c + j] * tw;
        }
    }
    Y[b * params.outputSize + o] = sum;
}
`
