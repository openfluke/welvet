package webgpu

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
	"github.com/openfluke/welvet/tiling"
)

type swiGLUParams struct {
	N    uint32
	Pad0 uint32
	Pad1 uint32
	Pad2 uint32
}

// SwiGLUFuse computes out[i] = silu(gate[i]) * up[i] elementwise on a real
// WebGPU device. No host fallback.
func SwiGLUFuse(gate, up, out []float32, n int) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu SwiGLUFuse: %w", initErr)
	}
	if len(gate) < n || len(up) < n || len(out) < n {
		return fmt.Errorf("webgpu SwiGLUFuse: shape")
	}
	if err := sess.ensureSwiGLUPipe(); err != nil {
		return err
	}
	return sess.swiGLUFuse(gate, up, out, n)
}

func (s *session) ensureSwiGLUPipe() error {
	if s.pipeSwiGLUFuse != nil {
		return nil
	}
	var err error
	s.pipeSwiGLUFuse, err = makePipeline(s.device, ShaderSwiGLUFuse, "welvet-swiglu-fuse")
	return err
}

func (s *session) swiGLUFuse(gate, up, out []float32, n int) error {
	const wg = 64
	dev, q := s.device, s.queue

	gateBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-swiglu-gate", Contents: wgpu.ToBytes(gate[:n]),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer gateBuf.Destroy()

	upBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-swiglu-up", Contents: wgpu.ToBytes(up[:n]),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer upBuf.Destroy()

	outBytes := uint64(n * 4)
	if outBytes < 64 {
		outBytes = 64
	}
	outBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "welvet-swiglu-out", Size: outBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer outBuf.Destroy()

	p := swiGLUParams{N: uint32(n)}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-swiglu-p", Contents: wgpu.ToBytes([]swiGLUParams{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()

	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.pipeSwiGLUFuse.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: gateBuf, Offset: 0, Size: gateBuf.GetSize()},
			{Binding: 2, Buffer: upBuf, Offset: 0, Size: upBuf.GetSize()},
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
	pass.SetPipeline(s.pipeSwiGLUFuse)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(tiling.GPUWorkgroupsX(n, wg), 1, 1)
	pass.End()
	cmd, err := enc.Finish(nil)
	if err != nil {
		return err
	}
	q.Submit(cmd)

	outData, err := readbackF32(dev, q, outBuf, n)
	if err != nil {
		return err
	}
	copy(out, outData)
	return nil
}

// ShaderSwiGLUFuse — out[i] = silu(gate[i]) * up[i], silu(x) = x*sigmoid(x).
const ShaderSwiGLUFuse = `
struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> Gate: array<f32>;
@group(0) @binding(2) var<storage, read> Up: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let g = Gate[i];
    let silu = g / (1.0 + exp(-g));
    Out[i] = silu * Up[i];
}
`
