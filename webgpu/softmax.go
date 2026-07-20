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

// Softmax runs the standard/grid/hierarchical softmax math on a real WebGPU
// device: one workgroup per group (row), reducing max then sum then normalizing.
// x/y are flattened [nGroups*size]. No host fallback.
func Softmax(x, y []float32, nGroups, size int, temp float32) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu Softmax: %w", initErr)
	}
	if len(x) < nGroups*size || len(y) < nGroups*size {
		return fmt.Errorf("webgpu Softmax: shape")
	}
	if temp == 0 {
		temp = 1.0
	}
	if err := sess.ensureSoftmaxPipes(); err != nil {
		return err
	}
	return sess.softmaxFwd(x, y, nGroups, size, temp)
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

func (s *session) softmaxFwd(x, y []float32, nGroups, size int, temp float32) error {
	dev, q := s.device, s.queue

	inBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-sm-in", Contents: wgpu.ToBytes(x[:nGroups*size]),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer inBuf.Destroy()

	outBytes := uint64(nGroups * size * 4)
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

	// softmaxType=0 (standard) never touches the mask buffer body; a minimal
	// dummy satisfies the binding.
	maskBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-sm-mask", Contents: wgpu.ToBytes([]uint32{0xFFFFFFFF}),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer maskBuf.Destroy()

	p := softmaxParams{BatchSize: uint32(nGroups), Size: uint32(size), Temp: temp, Type: 0}
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

	outY, err := readbackF32(dev, q, outBuf, nGroups*size)
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

// ShaderSoftmaxForward — ported from loom/poly/wgpu_softmax.go. Only softmaxType=0
// (standard) is exercised by welvet callers; other type codes are dead paths kept
// for shader-source parity with loom.
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

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let b = wg_id.x;
    let tid = local_id.x;
    let size = params.size;
    let base = b * size;
    var rng_seed = params.seed + b * 1337u + tid;

    var local_max: f32 = -1e38;
    for (var i = tid; i < size; i += 256u) {
        var val = input[base + i] / params.temp;

        if (params.softmaxType == 4u) {
            var u = rand_f32(&rng_seed);
            if (u < 1e-10) { u = 1e-10; }
            val = val - log(-log(u));
        }

        if (params.softmaxType == 5u) {
             let maskIdx = i / 32u;
             let maskBit = i % 32u;
             let is_masked = (mask[maskIdx] & (1u << maskBit)) == 0u;
             if (is_masked) { val = -1e30; }
        }

        local_max = max(local_max, val);
    }
    shared_reduce[tid] = local_max;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_reduce[tid] = max(shared_reduce[tid], shared_reduce[tid + s]);
        }
        workgroupBarrier();
    }
    let global_max = shared_reduce[0];
    workgroupBarrier();

    var local_sum: f32 = 0.0;
    for (var i = tid; i < size; i += 256u) {
        var val = input[base + i] / params.temp;
        if (params.softmaxType == 4u) {
             var sum_seed = params.seed + b * 1337u + tid;
             var u = rand_f32(&sum_seed);
             if (u < 1e-10) { u = 1e-10; }
             val = val - log(-log(u));
        }
        if (params.softmaxType == 5u) {
             let maskIdx = i / 32u;
             let maskBit = i % 32u;
             if ((mask[maskIdx] & (1u << maskBit)) == 0u) { val = -1e30; }
        }
        local_sum += exp(val - global_max);
    }
    shared_reduce[tid] = local_sum;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_reduce[tid] += shared_reduce[tid + s];
        }
        workgroupBarrier();
    }
    let global_sum = shared_reduce[0];
    workgroupBarrier();

    for (var i = tid; i < size; i += 256u) {
        var val = input[base + i] / params.temp;
        if (params.softmaxType == 4u) {
             var out_seed = params.seed + b * 1337u + tid;
             var u = rand_f32(&out_seed);
             if (u < 1e-10) { u = 1e-10; }
             val = val - log(-log(u));
        }
        if (params.softmaxType == 5u) {
             let maskIdx = i / 32u;
             let maskBit = i % 32u;
             if ((mask[maskIdx] & (1u << maskBit)) == 0u) { val = -1e30; }
        }
        output[base + i] = exp(val - global_max) / global_sum;
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
