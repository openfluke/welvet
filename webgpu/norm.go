package webgpu

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
)

// normParams mirrors the WGSL Params uniform for RMSNorm/LayerNorm forward.
type normParams struct {
	NTok uint32
	Dim  uint32
	Eps  float32
	Pad  uint32
}

// rmsBwdReduceParams mirrors the WGSL Params uniform for the dGamma token-reduce pass.
type rmsBwdReduceParams struct {
	NTok uint32
	Dim  uint32
	Pad0 uint32
	Pad1 uint32
}

// RMSNorm computes y[t,i] = x[t,i] * invRMS[t] * gamma[i] on a real WebGPU device,
// one token per workgroup (t = batch*seq flattened). No host fallback.
func RMSNorm(x, gamma, y []float32, nTok, dim int, eps float32) error {
	return RMSNormXHat(x, gamma, nil, y, nTok, dim, eps)
}

// RMSNormXHat is RMSNorm but additionally writes the pre-affine x̂ = x*invRMS into
// xHat (skipped when xHat is nil). Layers need x̂ as the Backward "pre" contract.
func RMSNormXHat(x, gamma, xHat, y []float32, nTok, dim int, eps float32) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu RMSNorm: %w", initErr)
	}
	if len(x) < nTok*dim || len(gamma) < dim || len(y) < nTok*dim {
		return fmt.Errorf("webgpu RMSNorm: shape")
	}
	if xHat != nil && len(xHat) < nTok*dim {
		return fmt.Errorf("webgpu RMSNorm: xHat shape")
	}
	if err := sess.ensureNormPipes(); err != nil {
		return err
	}
	return sess.rmsNormFwd(x, gamma, xHat, y, nTok, dim, eps)
}

// RMSNormBackward computes dx and dGamma on device from dy, x, xHat (pre), gamma.
func RMSNormBackward(dy, x, xHat, gamma, dx, dGamma []float32, nTok, dim int, eps float32) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu RMSNormBackward: %w", initErr)
	}
	if len(dy) < nTok*dim || len(x) < nTok*dim || len(xHat) < nTok*dim || len(gamma) < dim ||
		len(dx) < nTok*dim || len(dGamma) < dim {
		return fmt.Errorf("webgpu RMSNormBackward: shape")
	}
	if err := sess.ensureNormPipes(); err != nil {
		return err
	}
	return sess.rmsNormBwd(dy, x, xHat, gamma, dx, dGamma, nTok, dim, eps)
}

// LayerNorm computes y = ((x-mean)/sqrt(var+eps))*gamma + beta on a real WebGPU
// device, one token per workgroup. No host fallback.
func LayerNorm(x, gamma, beta, y []float32, nTok, dim int, eps float32) error {
	return LayerNormXHat(x, gamma, beta, nil, y, nTok, dim, eps)
}

// LayerNormXHat is LayerNorm but additionally writes the pre-affine x̂ into xHat
// (skipped when xHat is nil).
func LayerNormXHat(x, gamma, beta, xHat, y []float32, nTok, dim int, eps float32) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu LayerNorm: %w", initErr)
	}
	if len(x) < nTok*dim || len(gamma) < dim || len(beta) < dim || len(y) < nTok*dim {
		return fmt.Errorf("webgpu LayerNorm: shape")
	}
	if xHat != nil && len(xHat) < nTok*dim {
		return fmt.Errorf("webgpu LayerNorm: xHat shape")
	}
	if err := sess.ensureNormPipes(); err != nil {
		return err
	}
	return sess.layerNormFwd(x, gamma, beta, xHat, y, nTok, dim, eps)
}

func (s *session) ensureNormPipes() error {
	var err error
	if s.pipeRMSNormFwd == nil {
		if s.pipeRMSNormFwd, err = makePipeline(s.device, ShaderRMSNormForward, "welvet-rmsnorm-fwd"); err != nil {
			return err
		}
	}
	if s.pipeRMSNormBwd == nil {
		if s.pipeRMSNormBwd, err = makePipeline(s.device, ShaderRMSNormBackward, "welvet-rmsnorm-bwd"); err != nil {
			return err
		}
	}
	if s.pipeRMSNormBwdRed == nil {
		if s.pipeRMSNormBwdRed, err = makePipeline(s.device, ShaderRMSNormBackwardReduce, "welvet-rmsnorm-bwd-reduce"); err != nil {
			return err
		}
	}
	if s.pipeLayerNormFwd == nil {
		if s.pipeLayerNormFwd, err = makePipeline(s.device, ShaderLayerNormForward, "welvet-layernorm-fwd"); err != nil {
			return err
		}
	}
	return nil
}

func (s *session) rmsNormFwd(x, gamma, xHat, y []float32, nTok, dim int, eps float32) error {
	dev, q := s.device, s.queue

	xBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-rms-x", Contents: wgpu.ToBytes(x[:nTok*dim]),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer xBuf.Destroy()

	gBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-rms-g", Contents: wgpu.ToBytes(gamma[:dim]),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer gBuf.Destroy()

	yBytes := uint64(nTok * dim * 4)
	if yBytes < 64 {
		yBytes = 64
	}
	yBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "welvet-rms-y", Size: yBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer yBuf.Destroy()

	p := normParams{NTok: uint32(nTok), Dim: uint32(dim), Eps: eps}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-rms-p", Contents: wgpu.ToBytes([]normParams{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()

	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.pipeRMSNormFwd.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: xBuf, Offset: 0, Size: xBuf.GetSize()},
			{Binding: 2, Buffer: gBuf, Offset: 0, Size: gBuf.GetSize()},
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
	pass.SetPipeline(s.pipeRMSNormFwd)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(uint32(nTok), 1, 1)
	pass.End()
	cmd, err := enc.Finish(nil)
	if err != nil {
		return err
	}
	q.Submit(cmd)

	outY, err := readbackF32(dev, q, yBuf, nTok*dim)
	if err != nil {
		return err
	}
	copy(y, outY)
	if xHat != nil {
		// x̂ = y / gamma is unsafe when gamma has zeros; recompute on host cheaply
		// instead of adding a second device readback (x̂ is a pure elementwise
		// rescale of x already resident on host).
		var sumSq float64
		_ = sumSq
		for t := 0; t < nTok; t++ {
			base := t * dim
			var ss float64
			for i := 0; i < dim; i++ {
				v := float64(x[base+i])
				ss += v * v
			}
			inv := 1.0 / sqrt64(ss/float64(dim)+float64(eps))
			for i := 0; i < dim; i++ {
				xHat[base+i] = float32(float64(x[base+i]) * inv)
			}
		}
	}
	return nil
}

func (s *session) rmsNormBwd(dy, x, xHat, gamma, dx, dGamma []float32, nTok, dim int, eps float32) error {
	dev, q := s.device, s.queue

	mk := func(label string, data []float32) (*wgpu.Buffer, error) {
		return dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
			Label: label, Contents: wgpu.ToBytes(data),
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
		})
	}
	xBuf, err := mk("welvet-rmsb-x", x[:nTok*dim])
	if err != nil {
		return err
	}
	defer xBuf.Destroy()
	xhBuf, err := mk("welvet-rmsb-xhat", xHat[:nTok*dim])
	if err != nil {
		return err
	}
	defer xhBuf.Destroy()
	gBuf, err := mk("welvet-rmsb-g", gamma[:dim])
	if err != nil {
		return err
	}
	defer gBuf.Destroy()
	dyBuf, err := mk("welvet-rmsb-dy", dy[:nTok*dim])
	if err != nil {
		return err
	}
	defer dyBuf.Destroy()

	dxBytes := uint64(nTok * dim * 4)
	if dxBytes < 64 {
		dxBytes = 64
	}
	dxBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "welvet-rmsb-dx", Size: dxBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer dxBuf.Destroy()

	dgPartialBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "welvet-rmsb-dgpartial", Size: dxBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer dgPartialBuf.Destroy()

	p := normParams{NTok: uint32(nTok), Dim: uint32(dim), Eps: eps}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-rmsb-p", Contents: wgpu.ToBytes([]normParams{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()

	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.pipeRMSNormBwd.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: xBuf, Offset: 0, Size: xBuf.GetSize()},
			{Binding: 2, Buffer: xhBuf, Offset: 0, Size: xhBuf.GetSize()},
			{Binding: 3, Buffer: gBuf, Offset: 0, Size: gBuf.GetSize()},
			{Binding: 4, Buffer: dyBuf, Offset: 0, Size: dyBuf.GetSize()},
			{Binding: 5, Buffer: dxBuf, Offset: 0, Size: dxBuf.GetSize()},
			{Binding: 6, Buffer: dgPartialBuf, Offset: 0, Size: dgPartialBuf.GetSize()},
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
	pass.SetPipeline(s.pipeRMSNormBwd)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(uint32(nTok), 1, 1)
	pass.End()
	cmd, err := enc.Finish(nil)
	if err != nil {
		return err
	}
	q.Submit(cmd)

	outDx, err := readbackF32(dev, q, dxBuf, nTok*dim)
	if err != nil {
		return err
	}
	copy(dx, outDx)

	// Pass 2 — reduce per-token dGamma partials over the token axis on device.
	dgBytes := uint64(dim * 4)
	if dgBytes < 64 {
		dgBytes = 64
	}
	dgBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "welvet-rmsb-dg", Size: dgBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer dgBuf.Destroy()

	rp := rmsBwdReduceParams{NTok: uint32(nTok), Dim: uint32(dim)}
	rpBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-rmsb-rp", Contents: wgpu.ToBytes([]rmsBwdReduceParams{rp}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer rpBuf.Destroy()

	bg2, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.pipeRMSNormBwdRed.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: rpBuf, Offset: 0, Size: rpBuf.GetSize()},
			{Binding: 1, Buffer: dgPartialBuf, Offset: 0, Size: dgPartialBuf.GetSize()},
			{Binding: 2, Buffer: dgBuf, Offset: 0, Size: dgBuf.GetSize()},
		},
	})
	if err != nil {
		return err
	}
	defer bg2.Release()

	enc2, err := dev.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	pass2 := enc2.BeginComputePass(nil)
	pass2.SetPipeline(s.pipeRMSNormBwdRed)
	pass2.SetBindGroup(0, bg2, nil)
	const wg = 64
	pass2.DispatchWorkgroups((uint32(dim)+wg-1)/wg, 1, 1)
	pass2.End()
	cmd2, err := enc2.Finish(nil)
	if err != nil {
		return err
	}
	q.Submit(cmd2)

	outDg, err := readbackF32(dev, q, dgBuf, dim)
	if err != nil {
		return err
	}
	copy(dGamma, outDg)
	return nil
}

func (s *session) layerNormFwd(x, gamma, beta, xHat, y []float32, nTok, dim int, eps float32) error {
	dev, q := s.device, s.queue

	mk := func(label string, data []float32) (*wgpu.Buffer, error) {
		return dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
			Label: label, Contents: wgpu.ToBytes(data),
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
		})
	}
	xBuf, err := mk("welvet-ln-x", x[:nTok*dim])
	if err != nil {
		return err
	}
	defer xBuf.Destroy()
	gBuf, err := mk("welvet-ln-g", gamma[:dim])
	if err != nil {
		return err
	}
	defer gBuf.Destroy()
	bBuf, err := mk("welvet-ln-b", beta[:dim])
	if err != nil {
		return err
	}
	defer bBuf.Destroy()

	yBytes := uint64(nTok * dim * 4)
	if yBytes < 64 {
		yBytes = 64
	}
	yBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "welvet-ln-y", Size: yBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer yBuf.Destroy()

	p := normParams{NTok: uint32(nTok), Dim: uint32(dim), Eps: eps}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-ln-p", Contents: wgpu.ToBytes([]normParams{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()

	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.pipeLayerNormFwd.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: xBuf, Offset: 0, Size: xBuf.GetSize()},
			{Binding: 2, Buffer: gBuf, Offset: 0, Size: gBuf.GetSize()},
			{Binding: 3, Buffer: bBuf, Offset: 0, Size: bBuf.GetSize()},
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
	pass.SetPipeline(s.pipeLayerNormFwd)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(uint32(nTok), 1, 1)
	pass.End()
	cmd, err := enc.Finish(nil)
	if err != nil {
		return err
	}
	q.Submit(cmd)

	outY, err := readbackF32(dev, q, yBuf, nTok*dim)
	if err != nil {
		return err
	}
	copy(y, outY)
	if xHat != nil {
		for t := 0; t < nTok; t++ {
			base := t * dim
			var sum, sumSq float64
			for i := 0; i < dim; i++ {
				v := float64(x[base+i])
				sum += v
				sumSq += v * v
			}
			mean := sum / float64(dim)
			variance := sumSq/float64(dim) - mean*mean
			inv := 1.0 / sqrt64(variance+float64(eps))
			for i := 0; i < dim; i++ {
				xHat[base+i] = float32((float64(x[base+i]) - mean) * inv)
			}
		}
	}
	return nil
}

func sqrt64(v float64) float64 {
	// Avoid importing math just for Sqrt in this tiny host-side rescale helper.
	if v <= 0 {
		return 0
	}
	x := v
	for i := 0; i < 40; i++ {
		x = 0.5 * (x + v/x)
	}
	return x
}

// ShaderRMSNormForward — one workgroup (64 threads) per token; y = x*invRMS*gamma.
const ShaderRMSNormForward = `
struct Params { nTok: u32, dim: u32, eps: f32, _pad: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> Gamma: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let t = wg.x;
    if (t >= params.nTok) { return; }
    let tid = lid.x;
    let dim = params.dim;
    let base = t * dim;

    var local: f32 = 0.0;
    for (var i = tid; i < dim; i += 64u) {
        let v = X[base + i];
        local += v * v;
    }
    partial[tid] = local;
    workgroupBarrier();
    var stride = 32u;
    while (stride > 0u) {
        if (tid < stride) { partial[tid] += partial[tid + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inv = inverseSqrt(partial[0] / f32(dim) + params.eps);
    workgroupBarrier();
    for (var i = tid; i < dim; i += 64u) {
        Y[base + i] = X[base + i] * inv * Gamma[i];
    }
}
`

// ShaderRMSNormBackward — pass 1: per-token dx and per-token dGamma partials
// (dGammaPartial[t,i] = dy[t,i]*xHat[t,i], reduced over tokens in pass 2).
const ShaderRMSNormBackward = `
struct Params { nTok: u32, dim: u32, eps: f32, _pad: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> XHat: array<f32>;
@group(0) @binding(3) var<storage, read> Gamma: array<f32>;
@group(0) @binding(4) var<storage, read> Dy: array<f32>;
@group(0) @binding(5) var<storage, read_write> Dx: array<f32>;
@group(0) @binding(6) var<storage, read_write> DGammaPartial: array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let t = wg.x;
    if (t >= params.nTok) { return; }
    let tid = lid.x;
    let dim = params.dim;
    let base = t * dim;

    var localSq: f32 = 0.0;
    for (var i = tid; i < dim; i += 64u) {
        let v = X[base + i];
        localSq += v * v;
    }
    partial[tid] = localSq;
    workgroupBarrier();
    var stride = 32u;
    while (stride > 0u) {
        if (tid < stride) { partial[tid] += partial[tid + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inv = inverseSqrt(partial[0] / f32(dim) + params.eps);
    workgroupBarrier();

    var localUX: f32 = 0.0;
    for (var i = tid; i < dim; i += 64u) {
        let xh = XHat[base + i];
        let d = Dy[base + i];
        let u = Gamma[i] * d;
        localUX += u * xh;
        DGammaPartial[base + i] = d * xh;
    }
    partial[tid] = localUX;
    workgroupBarrier();
    stride = 32u;
    while (stride > 0u) {
        if (tid < stride) { partial[tid] += partial[tid + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let mean = partial[0] / f32(dim);
    workgroupBarrier();

    for (var i = tid; i < dim; i += 64u) {
        let xh = XHat[base + i];
        let d = Dy[base + i];
        let u = Gamma[i] * d;
        Dx[base + i] = inv * (u - xh * mean);
    }
}
`

// ShaderRMSNormBackwardReduce — pass 2: dGamma[i] = Σ_t dGammaPartial[t,i].
const ShaderRMSNormBackwardReduce = `
struct Params { nTok: u32, dim: u32, _p0: u32, _p1: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> DGammaPartial: array<f32>;
@group(0) @binding(2) var<storage, read_write> DGamma: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.dim) { return; }
    var sum: f32 = 0.0;
    for (var t: u32 = 0u; t < params.nTok; t++) {
        sum += DGammaPartial[t * params.dim + i];
    }
    DGamma[i] = sum;
}
`

// ShaderLayerNormForward — one workgroup (64 threads) per token;
// y = ((x-mean)*inv)*gamma + beta.
const ShaderLayerNormForward = `
struct Params { nTok: u32, dim: u32, eps: f32, _pad: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> Gamma: array<f32>;
@group(0) @binding(3) var<storage, read> Beta: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;

var<workgroup> partialSum: array<f32, 64>;
var<workgroup> partialSq: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let t = wg.x;
    if (t >= params.nTok) { return; }
    let tid = lid.x;
    let dim = params.dim;
    let base = t * dim;

    var s: f32 = 0.0;
    var sq: f32 = 0.0;
    for (var i = tid; i < dim; i += 64u) {
        let v = X[base + i];
        s += v;
        sq += v * v;
    }
    partialSum[tid] = s;
    partialSq[tid] = sq;
    workgroupBarrier();
    var stride = 32u;
    while (stride > 0u) {
        if (tid < stride) {
            partialSum[tid] += partialSum[tid + stride];
            partialSq[tid] += partialSq[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let mean = partialSum[0] / f32(dim);
    let variance = partialSq[0] / f32(dim) - mean * mean;
    let inv = inverseSqrt(variance + params.eps);
    workgroupBarrier();

    for (var i = tid; i < dim; i += 64u) {
        let xh = (X[base + i] - mean) * inv;
        Y[base + i] = xh * Gamma[i] + Beta[i];
    }
}
`
