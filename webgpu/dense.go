package webgpu

import (
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync"

	"github.com/openfluke/webgpu/wgpu"
	"github.com/openfluke/welvet/tiling"
)

var (
	mu      sync.Mutex
	tried   bool
	haveGPU bool
	initErr error
	sess    *session
)

type session struct {
	device       *wgpu.Device
	queue        *wgpu.Queue
	pipeFP32     *wgpu.ComputePipeline
	pipeQ4       *wgpu.ComputePipeline
	pipeI8       *wgpu.ComputePipeline
	pipeQ8       *wgpu.ComputePipeline
	pipeTernary  *wgpu.ComputePipeline
	pipeBinary   *wgpu.ComputePipeline
	pipeIQ       *wgpu.ComputePipeline
	pipeIQT      *wgpu.ComputePipeline
	pipeQ4T      *wgpu.ComputePipeline
	pipeQ8T      *wgpu.ComputePipeline
	pipeTernaryT *wgpu.ComputePipeline
	pipeBinaryT  *wgpu.ComputePipeline
	pipeQ41      *wgpu.ComputePipeline
	pipeQ5       *wgpu.ComputePipeline
	pipeK        *wgpu.ComputePipeline
	pipeKT       *wgpu.ComputePipeline
	name         string
}

// Available reports whether a real WebGPU device was acquired.
func Available() bool {
	ensure()
	return haveGPU
}

// AdapterName returns the bound adapter name (empty if none).
func AdapterName() string {
	ensure()
	if sess == nil {
		return ""
	}
	return sess.name
}

func ensure() {
	mu.Lock()
	defer mu.Unlock()
	if tried {
		return
	}
	tried = true
	s, err := newSession()
	if err != nil {
		haveGPU = false
		initErr = err
		return
	}
	sess = s
	haveGPU = true
}

func resolveBackends() *wgpu.InstanceDescriptor {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("WELVET_WGPU_BACKEND"))) {
	case "all":
		return nil
	case "dx12", "d3d12":
		return &wgpu.InstanceDescriptor{Backends: wgpu.InstanceBackendDX12}
	case "vulkan", "vk":
		return &wgpu.InstanceDescriptor{Backends: wgpu.InstanceBackendVulkan}
	case "metal":
		return &wgpu.InstanceDescriptor{Backends: wgpu.InstanceBackendMetal}
	case "gl", "opengl":
		return &wgpu.InstanceDescriptor{Backends: wgpu.InstanceBackendGL}
	}
	switch runtime.GOOS {
	case "darwin", "ios":
		return &wgpu.InstanceDescriptor{Backends: wgpu.InstanceBackendMetal}
	case "windows":
		return &wgpu.InstanceDescriptor{Backends: wgpu.InstanceBackendDX12}
	default:
		return &wgpu.InstanceDescriptor{Backends: wgpu.InstanceBackendVulkan}
	}
}

func makePipeline(dev *wgpu.Device, code, label string) (*wgpu.ComputePipeline, error) {
	mod, err := dev.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          label,
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: code},
	})
	if err != nil {
		return nil, err
	}
	defer mod.Release()
	return dev.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   label,
		Compute: wgpu.ProgrammableStageDescriptor{Module: mod, EntryPoint: "main"},
	})
}

func newSession() (*session, error) {
	instance := wgpu.CreateInstance(resolveBackends())
	if instance == nil {
		return nil, fmt.Errorf("webgpu: CreateInstance returned nil")
	}
	adapter, err := instance.RequestAdapter(&wgpu.RequestAdapterOptions{
		PowerPreference: wgpu.PowerPreferenceHighPerformance,
	})
	if err != nil {
		return nil, fmt.Errorf("webgpu: RequestAdapter: %w", err)
	}
	device, err := adapter.RequestDevice(nil)
	if err != nil {
		return nil, fmt.Errorf("webgpu: RequestDevice: %w", err)
	}
	pipeFP32, err := makePipeline(device, ShaderDenseFP32, "welvet-dense-fp32")
	if err != nil {
		return nil, fmt.Errorf("webgpu: FP32 pipeline: %w", err)
	}
	pipeQ4, err := makePipeline(device, ShaderDenseQ4Any, "welvet-dense-q4")
	if err != nil {
		return nil, fmt.Errorf("webgpu: Q4 pipeline: %w", err)
	}
	pipeI8, err := makePipeline(device, ShaderDenseI8, "welvet-dense-i8")
	if err != nil {
		return nil, fmt.Errorf("webgpu: I8 pipeline: %w", err)
	}
	info := adapter.GetInfo()
	return &session{
		device:   device,
		queue:    device.GetQueue(),
		pipeFP32: pipeFP32,
		pipeQ4:   pipeQ4,
		pipeI8:   pipeI8,
		name:     info.Name,
	}, nil
}

type gpuParams struct {
	Batch      uint32
	InputSize  uint32
	OutputSize uint32
	Pad        uint32
}

type gpuParamsI8 struct {
	Batch      uint32
	InputSize  uint32
	OutputSize uint32
	Pad        uint32
	Scale      float32
	_          [3]float32
}

// DenseGEMV computes y = W @ x on a real WebGPU device (FP32 SSBO). No host fallback.
func DenseGEMV(w []float32, x []float32, y []float32, batch, in, out int) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu DenseGEMV: %w", initErr)
	}
	if len(w) < out*in || len(x) < batch*in || len(y) < batch*out {
		return fmt.Errorf("webgpu: dense shape")
	}
	return sess.gemvFP32(w, x, y, batch, in, out)
}

// DenseGEMVQ4 — on-device Q4_0 dequant GEMV. scales[blocks], packed as u32 words (4 per block of 32).
func DenseGEMVQ4(scales []float32, packed []uint32, x, y []float32, batch, in, out int) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu DenseGEMVQ4: %w", initErr)
	}
	blocks := (out*in + 31) / 32
	if len(scales) < blocks || len(packed) < blocks*4 || len(x) < batch*in || len(y) < batch*out {
		return fmt.Errorf("webgpu DenseGEMVQ4: shape")
	}
	return sess.gemvQ4(scales, packed, x, y, batch, in, out)
}

// DenseGEMVI8 — on-device int8 dequant GEMV. weights packed as little-endian bytes in u32 words.
func DenseGEMVI8(weightsU32 []uint32, scale float32, x, y []float32, batch, in, out int) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu DenseGEMVI8: %w", initErr)
	}
	needWords := (out*in + 3) / 4
	if len(weightsU32) < needWords || len(x) < batch*in || len(y) < batch*out {
		return fmt.Errorf("webgpu DenseGEMVI8: shape")
	}
	return sess.gemvI8(weightsU32, scale, x, y, batch, in, out)
}

// DenseGEMVT: gx = W^T @ gy on device (FP32). No host fallback.
func DenseGEMVT(w []float32, gy, gx []float32, batch, in, out int) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu DenseGEMVT: %w", initErr)
	}
	wt := make([]float32, in*out)
	for o := 0; o < out; o++ {
		for i := 0; i < in; i++ {
			wt[i*out+o] = w[o*in+i]
		}
	}
	tmp := make([]float32, batch*in)
	if err := sess.gemvFP32(wt, gy, tmp, batch, out, in); err != nil {
		return err
	}
	copy(gx, tmp)
	return nil
}

func (s *session) gemvFP32(w, x, y []float32, batch, in, out int) error {
	const wg = 64
	dev, q := s.device, s.queue

	wBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-w", Contents: wgpu.ToBytes(w[:out*in]),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer wBuf.Destroy()

	xBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-x", Contents: wgpu.ToBytes(x[:batch*in]),
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
		Label: "welvet-y", Size: yBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer yBuf.Destroy()

	p := gpuParams{Batch: uint32(batch), InputSize: uint32(in), OutputSize: uint32(out)}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-p", Contents: wgpu.ToBytes([]gpuParams{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()

	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.pipeFP32.GetBindGroupLayout(0),
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

	enc, err := dev.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(s.pipeFP32)
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

func (s *session) gemvQ4(scales []float32, packed []uint32, x, y []float32, batch, in, out int) error {
	const wg = 64
	dev, q := s.device, s.queue
	blocks := (out*in + 31) / 32

	scBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-q4-sc", Contents: wgpu.ToBytes(scales[:blocks]),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer scBuf.Destroy()

	pkBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-q4-pk", Contents: wgpu.ToBytes(packed[:blocks*4]),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pkBuf.Destroy()

	xBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-q4-x", Contents: wgpu.ToBytes(x[:batch*in]),
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
		Label: "welvet-q4-y", Size: yBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer yBuf.Destroy()

	p := gpuParams{Batch: uint32(batch), InputSize: uint32(in), OutputSize: uint32(out)}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-q4-p", Contents: wgpu.ToBytes([]gpuParams{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()

	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.pipeQ4.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: xBuf, Offset: 0, Size: xBuf.GetSize()},
			{Binding: 2, Buffer: scBuf, Offset: 0, Size: scBuf.GetSize()},
			{Binding: 3, Buffer: pkBuf, Offset: 0, Size: pkBuf.GetSize()},
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
	pass.SetPipeline(s.pipeQ4)
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

func (s *session) gemvI8(weightsU32 []uint32, scale float32, x, y []float32, batch, in, out int) error {
	const wg = 64
	dev, q := s.device, s.queue
	needWords := (out*in + 3) / 4

	wBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-i8-w", Contents: wgpu.ToBytes(weightsU32[:needWords]),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer wBuf.Destroy()

	xBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-i8-x", Contents: wgpu.ToBytes(x[:batch*in]),
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
		Label: "welvet-i8-y", Size: yBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer yBuf.Destroy()

	p := gpuParamsI8{
		Batch: uint32(batch), InputSize: uint32(in), OutputSize: uint32(out), Scale: scale,
	}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-i8-p", Contents: wgpu.ToBytes([]gpuParamsI8{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()

	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.pipeI8.GetBindGroupLayout(0),
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

	enc, err := dev.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(s.pipeI8)
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

func readbackF32(dev *wgpu.Device, q *wgpu.Queue, buf *wgpu.Buffer, count int) ([]float32, error) {
	size := uint64(count * 4)
	if size < 64 {
		size = 64
	}
	staging, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "welvet-readback",
		Size:  size,
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, err
	}
	defer staging.Destroy()

	enc, err := dev.CreateCommandEncoder(nil)
	if err != nil {
		return nil, err
	}
	enc.CopyBufferToBuffer(buf, 0, staging, 0, size)
	cmd, err := enc.Finish(nil)
	if err != nil {
		return nil, err
	}
	q.Submit(cmd)

	done := make(chan struct{})
	if err := staging.MapAsync(wgpu.MapModeRead, 0, size, func(status wgpu.BufferMapAsyncStatus) {
		close(done)
	}); err != nil {
		return nil, err
	}
	for {
		dev.Poll(false, nil)
		select {
		case <-done:
			data := staging.GetMappedRange(0, uint(size))
			defer staging.Unmap()
			out := make([]float32, count)
			copy(wgpu.ToBytes(out), data[:count*4])
			return out, nil
		default:
		}
	}
}

// ShaderDenseFP32 — FP32 tiled GEMV.
const ShaderDenseFP32 = `
struct Params { batch: u32, inputSize: u32, outputSize: u32, _pad: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    let b = gid.y;
    if (o >= params.outputSize || b >= params.batch) { return; }
    var sum: f32 = 0.0;
    let xBase = b * params.inputSize;
    let wBase = o * params.inputSize;
    for (var i: u32 = 0u; i < params.inputSize; i++) {
        sum += X[xBase + i] * W[wBase + i];
    }
    Y[b * params.outputSize + o] = sum;
}
`

// ShaderDenseQ4 — on-device Q4_0 dequant (block=32, 1 scale + 4×u32 packed).
const ShaderDenseQ4 = `
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
    let nBlocks = params.inputSize / 32u;
    for (var blk: u32 = 0u; blk < nBlocks; blk++) {
        let globalIdx = wBase + blk * 32u;
        let blockIdx = globalIdx / 32u;
        let scale = scales[blockIdx];
        let wWord = (globalIdx / 8u);
        let i0 = xBase + blk * 32u;
        for (var w: u32 = 0u; w < 4u; w++) {
            let packed = weights[wWord + w];
            let base = i0 + w * 8u;
            var acc: f32 = 0.0;
            for (var n: u32 = 0u; n < 8u; n++) {
                var q = i32((packed >> (n * 4u)) & 0xFu);
                if (q > 7) { q -= 16; }
                acc += X[base + n] * f32(q);
            }
            sum += acc * scale;
        }
    }
    Y[b * params.outputSize + o] = sum;
}
`

// ShaderDenseI8 — on-device int8 × f32 acts, then * scale.
const ShaderDenseI8 = `
struct Params {
    batch: u32,
    inputSize: u32,
    outputSize: u32,
    _pad: u32,
    scale: f32,
    _p1: f32,
    _p2: f32,
    _p3: f32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    let b = gid.y;
    if (o >= params.outputSize || b >= params.batch) { return; }
    var sum: f32 = 0.0;
    let xBase = b * params.inputSize;
    let wBase = o * params.inputSize;
    let limit = params.inputSize / 4u;
    var i: u32 = 0u;
    for (var k: u32 = 0u; k < limit; k++) {
        let packed = W[(wBase + i) / 4u];
        var q0 = i32(packed & 0xFFu); if (q0 > 127) { q0 -= 256; }
        var q1 = i32((packed >> 8u) & 0xFFu); if (q1 > 127) { q1 -= 256; }
        var q2 = i32((packed >> 16u) & 0xFFu); if (q2 > 127) { q2 -= 256; }
        var q3 = i32((packed >> 24u) & 0xFFu); if (q3 > 127) { q3 -= 256; }
        sum += X[xBase+i]*f32(q0) + X[xBase+i+1u]*f32(q1) + X[xBase+i+2u]*f32(q2) + X[xBase+i+3u]*f32(q3);
        i += 4u;
    }
    for (; i < params.inputSize; i++) {
        let packed = W[(wBase + i) / 4u];
        let byteIdx = (wBase + i) % 4u;
        var q = i32((packed >> (byteIdx * 8u)) & 0xFFu);
        if (q > 127) { q -= 256; }
        sum += X[xBase + i] * f32(q);
    }
    Y[b * params.outputSize + o] = sum * params.scale;
}
`
