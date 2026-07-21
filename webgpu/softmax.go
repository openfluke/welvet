package webgpu

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
)

// softmaxParams mirrors loom's WGPUSoftmaxParams uniform layout.
type softmaxParams struct {
	BatchSize   uint32
	Size        uint32
	Temp        float32
	Type        uint32
	Rows        uint32
	Cols        uint32
	Seed        uint32
	EntmaxAlpha float32
}

// Softmax GPU kind type codes (loom/poly SoftmaxType parity).
const (
	SoftmaxTypeStandard     uint32 = 0
	SoftmaxTypeGrid         uint32 = 1
	SoftmaxTypeHierarchical uint32 = 2
	SoftmaxTypeTemperature  uint32 = 3
	SoftmaxTypeGumbel       uint32 = 4
	SoftmaxTypeMasked       uint32 = 5
	SoftmaxTypeSparse       uint32 = 6
	SoftmaxTypeEntmax       uint32 = 9
)

// Softmax runs the standard/grid/hierarchical softmax math on a real WebGPU
// device: one workgroup per group (row), reducing max then sum then normalizing.
// x/y are flattened [nGroups*size]. No host fallback.
func Softmax(x, y []float32, nGroups, size int, temp float32) error {
	return SoftmaxEx(x, y, nil, nGroups, size, temp, SoftmaxTypeStandard, 0, 0)
}

// SoftmaxEx runs exotic or standard softmax on device. mask is optional per-element
// keep weights (>=0.5 unmasked); nil means all positions active. kind uses
// SoftmaxType* constants. No host fallback.
func SoftmaxEx(x, y, mask []float32, nGroups, size int, temp float32, kind uint32, seed uint32, entmaxAlpha float32) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu SoftmaxEx: %w", initErr)
	}
	n := nGroups * size
	if len(x) < n || len(y) < n {
		return fmt.Errorf("webgpu SoftmaxEx: shape")
	}
	if temp == 0 {
		temp = 1.0
	}
	kindType := kind
	if kind == SoftmaxTypeEntmax {
		if entmaxAlpha <= 1.0 {
			kindType = SoftmaxTypeStandard
		} else if entmaxAlpha >= 2.0 {
			kindType = SoftmaxTypeSparse
		}
	}
	if err := sess.ensureSoftmaxPipes(); err != nil {
		return err
	}
	return sess.softmaxFwdEx(x, y, mask, nGroups, size, temp, kindType, seed, entmaxAlpha)
}

// SoftmaxBackward computes gx = (y/temp) * (gy - dot(gy, y)) on device, one
// workgroup per group (row). y is the forward softmax output (per the layer's
// Backward "pre" contract). No host fallback.
func SoftmaxBackward(gy, y, gx []float32, nGroups, size int, temp float32) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu SoftmaxBackward: %w", initErr)
	}
	if len(gy) < nGroups*size || len(y) < nGroups*size || len(gx) < nGroups*size {
		return fmt.Errorf("webgpu SoftmaxBackward: shape")
	}
	if temp == 0 {
		temp = 1.0
	}
	if err := sess.ensureSoftmaxPipes(); err != nil {
		return err
	}
	return sess.softmaxBwd(gy, y, gx, nGroups, size, temp)
}

func (s *session) ensureSoftmaxPipes() error {
	var err error
	if s.pipeSoftmaxFwd == nil {
		if s.pipeSoftmaxFwd, err = makePipeline(s.device, ShaderSoftmaxForward, "welvet-softmax-fwd"); err != nil {
			return err
		}
	}
	if s.pipeSoftmaxBwd == nil {
		if s.pipeSoftmaxBwd, err = makePipeline(s.device, ShaderSoftmaxBackward, "welvet-softmax-bwd"); err != nil {
			return err
		}
	}
	return nil
}

func packSoftmaxMaskF32(mask []float32, n int) []uint32 {
	nWords := (n + 31) / 32
	if nWords == 0 {
		nWords = 1
	}
	out := make([]uint32, nWords)
	if len(mask) == 0 {
		for i := range out {
			out[i] = 0xFFFFFFFF
		}
		return out
	}
	for i := 0; i < n; i++ {
		if i < len(mask) && mask[i] >= 0.5 {
			out[i/32] |= 1 << (i % 32)
		}
	}
	return out
}

func (s *session) softmaxFwdEx(x, y, mask []float32, nGroups, size int, temp float32, kind uint32, seed uint32, entmaxAlpha float32) error {
	dev, q := s.device, s.queue
	n := nGroups * size

	inBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-sm-in", Contents: wgpu.ToBytes(x[:n]),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer inBuf.Destroy()

	outBytes := uint64(n * 4)
	if outBytes < 64 {
		outBytes = 64
	}
	outBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "welvet-sm-out", Size: outBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer outBuf.Destroy()

	maskWords := packSoftmaxMaskF32(mask, n)
	maskBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-sm-mask", Contents: wgpu.ToBytes(maskWords),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer maskBuf.Destroy()

	p := softmaxParams{
		BatchSize: uint32(nGroups), Size: uint32(size), Temp: temp,
		Type: kind, Seed: seed, EntmaxAlpha: entmaxAlpha,
	}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-sm-p", Contents: wgpu.ToBytes([]softmaxParams{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()

	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.pipeSoftmaxFwd.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: inBuf, Offset: 0, Size: inBuf.GetSize()},
			{Binding: 2, Buffer: outBuf, Offset: 0, Size: outBuf.GetSize()},
			{Binding: 3, Buffer: maskBuf, Offset: 0, Size: maskBuf.GetSize()},
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
	pass.SetPipeline(s.pipeSoftmaxFwd)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(uint32(nGroups), 1, 1)
	pass.End()
	cmd, err := enc.Finish(nil)
	if err != nil {
		return err
	}
	q.Submit(cmd)

	outY, err := readbackF32(dev, q, outBuf, n)
	if err != nil {
		return err
	}
	copy(y, outY)
	return nil
}

func (s *session) softmaxBwd(gy, y, gx []float32, nGroups, size int, temp float32) error {
	dev, q := s.device, s.queue

	mk := func(label string, data []float32) (*wgpu.Buffer, error) {
		return dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
			Label: label, Contents: wgpu.ToBytes(data),
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
		})
	}
	gyBuf, err := mk("welvet-smb-gy", gy[:nGroups*size])
	if err != nil {
		return err
	}
	defer gyBuf.Destroy()
	yBuf, err := mk("welvet-smb-y", y[:nGroups*size])
	if err != nil {
		return err
	}
	defer yBuf.Destroy()

	gxBytes := uint64(nGroups * size * 4)
	if gxBytes < 64 {
		gxBytes = 64
	}
	gxBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "welvet-smb-gx", Size: gxBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer gxBuf.Destroy()

	p := softmaxParams{BatchSize: uint32(nGroups), Size: uint32(size), Temp: temp}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-smb-p", Contents: wgpu.ToBytes([]softmaxParams{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()

	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.pipeSoftmaxBwd.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: gyBuf, Offset: 0, Size: gyBuf.GetSize()},
			{Binding: 2, Buffer: yBuf, Offset: 0, Size: yBuf.GetSize()},
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
	pass.SetPipeline(s.pipeSoftmaxBwd)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(uint32(nGroups), 1, 1)
	pass.End()
	cmd, err := enc.Finish(nil)
	if err != nil {
		return err
	}
	q.Submit(cmd)

	outGx, err := readbackF32(dev, q, gxBuf, nGroups*size)
	if err != nil {
		return err
	}
	copy(gx, outGx)
	return nil
}

// ShaderSoftmaxForward — loom/poly parity with sparsemax (6) and entmax (9).
const ShaderSoftmaxForward = `
struct Params {
    batchSize: u32,
    size: u32,
    temp: f32,
    softmaxType: u32,
    rows: u32,
    cols: u32,
    seed: u32,
    entmaxAlpha: f32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> mask: array<u32>;

var<workgroup> shared_reduce: array<f32, 256>;

fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f32(seed: ptr<function, u32>) -> f32 {
    *seed = pcg_hash(*seed);
    return f32(*seed) / 4294967296.0;
}

fn is_masked(global_idx: u32) -> bool {
    let maskIdx = global_idx / 32u;
    let maskBit = global_idx % 32u;
    return (mask[maskIdx] & (1u << maskBit)) == 0u;
}

fn raw_scaled(base: u32, i: u32) -> f32 {
    return input[base + i] / params.temp;
}

fn scaled_logit(base: u32, i: u32, b: u32, tid: u32, phase: u32) -> f32 {
    var val = input[base + i] / params.temp;
    if (params.softmaxType == 4u) {
        var seed = params.seed + b * 1337u + tid + phase * 7919u;
        var u = rand_f32(&seed);
        if (u < 1e-10) { u = 1e-10; }
        val = val - log(-log(u));
    }
    if (params.softmaxType == 5u) {
        if (is_masked(base + i)) { val = -1e30; }
    }
    return val;
}

fn wg_reduce(op: u32, tid: u32) {
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            if (op == 0u) {
                shared_reduce[tid] = max(shared_reduce[tid], shared_reduce[tid + s]);
            } else if (op == 1u) {
                shared_reduce[tid] += shared_reduce[tid + s];
            } else {
                shared_reduce[tid] = min(shared_reduce[tid], shared_reduce[tid + s]);
            }
        }
        workgroupBarrier();
    }
}

fn sparsemax_tau(base: u32, size: u32, tid: u32) -> f32 {
    var local_min: f32 = 1e38;
    var local_max: f32 = -1e38;
    for (var i = tid; i < size; i += 256u) {
        let v = raw_scaled(base, i);
        local_min = min(local_min, v);
        local_max = max(local_max, v);
    }
    shared_reduce[tid] = local_min;
    workgroupBarrier();
    wg_reduce(2u, tid);
    let gmin = shared_reduce[0];
    workgroupBarrier();

    shared_reduce[tid] = local_max;
    workgroupBarrier();
    wg_reduce(0u, tid);
    let gmax = shared_reduce[0];
    workgroupBarrier();

    var lo = gmin - 1.0;
    var hi = gmax;
    for (var iter = 0u; iter < 48u; iter++) {
        let mid = (lo + hi) * 0.5;
        var local_sum: f32 = 0.0;
        for (var i = tid; i < size; i += 256u) {
            local_sum += max(0.0, raw_scaled(base, i) - mid);
        }
        shared_reduce[tid] = local_sum;
        workgroupBarrier();
        wg_reduce(1u, tid);
        let gs = shared_reduce[0];
        workgroupBarrier();
        if (gs > 1.0) {
            lo = mid;
        } else {
            hi = mid;
        }
        workgroupBarrier();
    }
    return (lo + hi) * 0.5;
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let b = wg_id.x;
    let tid = local_id.x;
    let size = params.size;
    let base = b * size;

    if (params.softmaxType == 6u) {
        let tau = sparsemax_tau(base, size, tid);
        for (var i = tid; i < size; i += 256u) {
            output[base + i] = max(0.0, raw_scaled(base, i) - tau);
        }
        return;
    }

    var local_max: f32 = -1e38;
    for (var i = tid; i < size; i += 256u) {
        local_max = max(local_max, scaled_logit(base, i, b, tid, 0u));
    }
    shared_reduce[tid] = local_max;
    workgroupBarrier();
    wg_reduce(0u, tid);
    let global_max = shared_reduce[0];
    workgroupBarrier();

    var local_sum: f32 = 0.0;
    for (var i = tid; i < size; i += 256u) {
        local_sum += exp(scaled_logit(base, i, b, tid, 1u) - global_max);
    }
    shared_reduce[tid] = local_sum;
    workgroupBarrier();
    wg_reduce(1u, tid);
    let global_sum = shared_reduce[0];
    workgroupBarrier();

    for (var i = tid; i < size; i += 256u) {
        let val = scaled_logit(base, i, b, tid, 2u);
        output[base + i] = exp(val - global_max) / global_sum;
    }

    if (params.softmaxType == 9u) {
        workgroupBarrier();
        let w = params.entmaxAlpha - 1.0;
        let tau = sparsemax_tau(base, size, tid);
        var local_blend_sum: f32 = 0.0;
        for (var i = tid; i < size; i += 256u) {
            let soft = output[base + i];
            let sparse = max(0.0, raw_scaled(base, i) - tau);
            let blended = (1.0 - w) * soft + w * sparse;
            output[base + i] = blended;
            local_blend_sum += blended;
        }
        shared_reduce[tid] = local_blend_sum;
        workgroupBarrier();
        wg_reduce(1u, tid);
        let blend_sum = shared_reduce[0];
        workgroupBarrier();
        if (blend_sum > 0.0) {
            for (var i = tid; i < size; i += 256u) {
                output[base + i] = output[base + i] / blend_sum;
            }
        }
    }
}
`

// ShaderSoftmaxBackward — ported from loom/poly/wgpu_softmax.go.
const ShaderSoftmaxBackward = `
struct Params {
    batchSize: u32,
    size: u32,
    temp: f32,
    softmaxType: u32,
    rows: u32,
    cols: u32,
    seed: u32,
    entmaxAlpha: f32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gradOutput: array<f32>;
@group(0) @binding(2) var<storage, read> softmaxOutput: array<f32>;
@group(0) @binding(3) var<storage, read_write> gradInput: array<f32>;

var<workgroup> shared_reduce: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let b = wg_id.x;
    let tid = local_id.x;
    let size = params.size;
    let base = b * size;

    var local_dot: f32 = 0.0;
    for (var i = tid; i < size; i += 256u) {
        local_dot += gradOutput[base + i] * softmaxOutput[base + i];
    }
    shared_reduce[tid] = local_dot;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_reduce[tid] += shared_reduce[tid + s];
        }
        workgroupBarrier();
    }
    let dot_prod = shared_reduce[0];
    workgroupBarrier();

    for (var i = tid; i < size; i += 256u) {
        let p = softmaxOutput[base + i];
        gradInput[base + i] = (p / params.temp) * (gradOutput[base + i] - dot_prod);
    }
}
`
