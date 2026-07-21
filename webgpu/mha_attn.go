package webgpu

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
)

// RoPEApply rotates Q or K pairs in-place on a real WebGPU device.
//
// Buffer layout (row-major f32):
//
//	index(s,h,d) = (s*numHeads + h)*headDim + d
//
// i.e. seqLen tokens, each numHeads*headDim floats (heads packed contiguously).
//
// positions selects absolute token positions:
//   - len 0 or 1: positions[0] (or 0) is the absolute position of s=0; token s uses offset+s
//   - len seqLen: positions[s] is the absolute position for each token (one dispatch each)
func RoPEApply(data []float32, seqLen, numHeads, headDim int, theta float32, positions []int32) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu RoPEApply: %w", initErr)
	}
	n := seqLen * numHeads * headDim
	if len(data) < n || seqLen <= 0 || numHeads <= 0 || headDim <= 0 || headDim%2 != 0 {
		return fmt.Errorf("webgpu RoPEApply: shape")
	}
	if err := sess.ensureRoPEPipe(); err != nil {
		return err
	}
	switch len(positions) {
	case 0:
		return sess.ropeApply(data[:n], seqLen, numHeads, headDim, theta, 0)
	case 1:
		return sess.ropeApply(data[:n], seqLen, numHeads, headDim, theta, int(positions[0]))
	default:
		if len(positions) != seqLen {
			return fmt.Errorf("webgpu RoPEApply: positions len %d != seqLen %d", len(positions), seqLen)
		}
		stride := numHeads * headDim
		for s := 0; s < seqLen; s++ {
			if err := sess.ropeApply(data[s*stride:(s+1)*stride], 1, numHeads, headDim, theta, int(positions[s])); err != nil {
				return err
			}
		}
		return nil
	}
}

// MHAConfig holds tiled multi-head attention dispatch parameters.
type MHAConfig struct {
	NumHeads, NumKVHeads, HeadDim int
	SeqLen, KVOffset, MaxSeqLen   int
	KvLen                         int // total key length (bidirectional); ignored when Causal
	TileSize                      int
	Causal                        bool
}

// MHAForward runs tiled softmax attention on device.
//
// q layout: [seqLen, numHeads, headDim] — same as RoPEApply.
// kCache/vCache layout: [numKVHeads, maxSeqLen, headDim] — index (h*maxSeqLen+pos)*headDim+d.
// out layout: [seqLen, numHeads, headDim] (same as q).
//
// Causal: query s at absolute position KVOffset+s attends to keys 0..KVOffset+s.
// Bidirectional: every query attends to keys 0..KvLen-1 (KvLen must be set).
func MHAForward(q, kCache, vCache, out []float32, cfg MHAConfig) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu MHAForward: %w", initErr)
	}
	if cfg.TileSize <= 0 {
		cfg.TileSize = 32
	}
	qN := cfg.SeqLen * cfg.NumHeads * cfg.HeadDim
	cacheN := cfg.NumKVHeads * cfg.MaxSeqLen * cfg.HeadDim
	if len(q) < qN || len(kCache) < cacheN || len(vCache) < cacheN || len(out) < qN {
		return fmt.Errorf("webgpu MHAForward: shape")
	}
	if cfg.NumHeads <= 0 || cfg.NumKVHeads <= 0 || cfg.HeadDim <= 0 || cfg.NumHeads%cfg.NumKVHeads != 0 {
		return fmt.Errorf("webgpu MHAForward: head geometry")
	}
	if !cfg.Causal && cfg.KvLen <= 0 {
		return fmt.Errorf("webgpu MHAForward: bidirectional needs KvLen>0")
	}
	pipe, err := sess.ensureMHAPipe(cfg.TileSize, cfg.HeadDim)
	if err != nil {
		return err
	}
	return sess.mhaForward(q[:qN], kCache[:cacheN], vCache[:cacheN], out[:qN], cfg, pipe)
}

// KVCacheUpdate writes newK/newV into kCache/vCache at absolute offset.
//
// newK/newV layout: [numTokens, numKVHeads*headDim] row-major (dense K/V projection layout).
// cache layout: [numKVHeads, maxSeqLen, headDim] (same as MHAForward).
func KVCacheUpdate(kCache, vCache, newK, newV []float32, offset, headDim, maxSeqLen, numKVHeads, numTokens int) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu KVCacheUpdate: %w", initErr)
	}
	kvDim := numKVHeads * headDim
	cacheN := numKVHeads * maxSeqLen * headDim
	need := numTokens * kvDim
	if len(newK) < need || len(newV) < need || len(kCache) < cacheN || len(vCache) < cacheN {
		return fmt.Errorf("webgpu KVCacheUpdate: shape")
	}
	if err := sess.ensureKVUpdatePipe(); err != nil {
		return err
	}
	return sess.kvCacheUpdate(kCache[:cacheN], vCache[:cacheN], newK[:need], newV[:need],
		offset, headDim, maxSeqLen, numKVHeads, numTokens)
}

// PackKVCache converts welvet host ring cache [maxSeqLen, kvDim] into GPU [numKVHeads, maxSeqLen, headDim].
// kvLen is the number of absolute key positions 0..kvLen-1 to copy.
func PackKVCache(host []float64, numKVHeads, maxSeqLen, headDim, kvLen int) []float32 {
	kvDim := numKVHeads * headDim
	out := make([]float32, numKVHeads*maxSeqLen*headDim)
	if kvLen > maxSeqLen {
		kvLen = maxSeqLen
	}
	for kPos := 0; kPos < kvLen; kPos++ {
		src := host[(kPos%maxSeqLen)*kvDim : (kPos%maxSeqLen+1)*kvDim]
		for h := 0; h < numKVHeads; h++ {
			for d := 0; d < headDim; d++ {
				out[(h*maxSeqLen+kPos)*headDim+d] = float32(src[h*headDim+d])
			}
		}
	}
	return out
}

type ropeParams struct {
	SeqLen   uint32
	HeadDim  uint32
	NumHeads uint32
	Offset   uint32
	Theta    float32
	_        [3]uint32
}

type mhaParams struct {
	NumHeads   uint32
	NumKVHeads uint32
	HeadDim    uint32
	SeqLen     uint32
	KVOffset   uint32
	MaxSeqLen  uint32
	TileSize   uint32
	Causal     uint32
	KvLen      uint32
}

type kvUpdateParams struct {
	Offset     uint32
	HeadDim    uint32
	MaxSeqLen  uint32
	NumKVHeads uint32
	NumTokens  uint32
	_          [7]uint32
}

func (s *session) ensureRoPEPipe() error {
	if s.pipeRoPE != nil {
		return nil
	}
	var err error
	s.pipeRoPE, err = makePipeline(s.device, ShaderRoPE, "welvet-rope")
	return err
}

func (s *session) ensureKVUpdatePipe() error {
	if s.pipeKVUpdate != nil {
		return nil
	}
	var err error
	s.pipeKVUpdate, err = makePipeline(s.device, ShaderKVUpdate, "welvet-kv-update")
	return err
}

func (s *session) ensureMHAPipe(tileSize, headDim int) (*wgpu.ComputePipeline, error) {
	if s.mhaPipes == nil {
		s.mhaPipes = make(map[uint64]*wgpu.ComputePipeline)
	}
	key := uint64(headDim)<<32 | uint64(tileSize)
	if p, ok := s.mhaPipes[key]; ok {
		return p, nil
	}
	code := shaderTiledMHAN(tileSize, headDim)
	p, err := makePipeline(s.device, code, fmt.Sprintf("welvet-mha-%d-%d", tileSize, headDim))
	if err != nil {
		return nil, err
	}
	s.mhaPipes[key] = p
	return p, nil
}

func (s *session) ropeApply(data []float32, seqLen, numHeads, headDim int, theta float32, offset int) error {
	dev, q := s.device, s.queue
	dataBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-rope-data", Contents: wgpu.ToBytes(data),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}
	defer dataBuf.Destroy()

	p := ropeParams{
		SeqLen: uint32(seqLen), HeadDim: uint32(headDim),
		NumHeads: uint32(numHeads), Offset: uint32(offset), Theta: theta,
	}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-rope-p", Contents: wgpu.ToBytes([]ropeParams{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()

	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.pipeRoPE.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: dataBuf, Offset: 0, Size: dataBuf.GetSize()},
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
	pass.SetPipeline(s.pipeRoPE)
	pass.SetBindGroup(0, bg, nil)
	halfDim := headDim / 2
	totalPairs := seqLen * numHeads * halfDim
	pass.DispatchWorkgroups((uint32(totalPairs)+63)/64, 1, 1)
	pass.End()
	cmd, err := enc.Finish(nil)
	if err != nil {
		return err
	}
	q.Submit(cmd)

	out, err := readbackF32(dev, q, dataBuf, len(data))
	if err != nil {
		return err
	}
	copy(data, out)
	return nil
}

func (s *session) kvCacheUpdate(kCache, vCache, newK, newV []float32, offset, headDim, maxSeqLen, numKVHeads, numTokens int) error {
	dev, q := s.device, s.queue
	mk := func(label string, data []float32) (*wgpu.Buffer, error) {
		return dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
			Label: label, Contents: wgpu.ToBytes(data),
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
		})
	}
	kcBuf, err := mk("welvet-kvc-kc", kCache)
	if err != nil {
		return err
	}
	defer kcBuf.Destroy()
	vcBuf, err := mk("welvet-kvc-vc", vCache)
	if err != nil {
		return err
	}
	defer vcBuf.Destroy()
	nkBuf, err := mk("welvet-kvc-nk", newK)
	if err != nil {
		return err
	}
	defer nkBuf.Destroy()
	nvBuf, err := mk("welvet-kvc-nv", newV)
	if err != nil {
		return err
	}
	defer nvBuf.Destroy()

	p := kvUpdateParams{
		Offset: uint32(offset), HeadDim: uint32(headDim), MaxSeqLen: uint32(maxSeqLen),
		NumKVHeads: uint32(numKVHeads), NumTokens: uint32(numTokens),
	}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-kvc-p", Contents: wgpu.ToBytes([]kvUpdateParams{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()

	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.pipeKVUpdate.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: kcBuf, Offset: 0, Size: kcBuf.GetSize()},
			{Binding: 1, Buffer: vcBuf, Offset: 0, Size: vcBuf.GetSize()},
			{Binding: 2, Buffer: nkBuf, Offset: 0, Size: nkBuf.GetSize()},
			{Binding: 3, Buffer: nvBuf, Offset: 0, Size: nvBuf.GetSize()},
			{Binding: 4, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
		},
	})
	if err != nil {
		return err
	}
	defer bg.Release()

	kvDim := numKVHeads * headDim
	enc, err := dev.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(s.pipeKVUpdate)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups((uint32(kvDim*numTokens)+63)/64, 1, 1)
	pass.End()
	cmd, err := enc.Finish(nil)
	if err != nil {
		return err
	}
	q.Submit(cmd)

	outK, err := readbackF32(dev, q, kcBuf, len(kCache))
	if err != nil {
		return err
	}
	copy(kCache, outK)
	outV, err := readbackF32(dev, q, vcBuf, len(vCache))
	if err != nil {
		return err
	}
	copy(vCache, outV)
	return nil
}

func (s *session) mhaForward(q, kCache, vCache, out []float32, cfg MHAConfig, pipe *wgpu.ComputePipeline) error {
	dev, qDev := s.device, s.queue
	mk := func(label string, data []float32) (*wgpu.Buffer, error) {
		return dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
			Label: label, Contents: wgpu.ToBytes(data),
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
		})
	}
	qBuf, err := mk("welvet-mha-q", q)
	if err != nil {
		return err
	}
	defer qBuf.Destroy()
	kBuf, err := mk("welvet-mha-kc", kCache)
	if err != nil {
		return err
	}
	defer kBuf.Destroy()
	vBuf, err := mk("welvet-mha-vc", vCache)
	if err != nil {
		return err
	}
	defer vBuf.Destroy()

	outN := len(out)
	outBytes := uint64(outN * 4)
	if outBytes < 64 {
		outBytes = 64
	}
	oBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "welvet-mha-out", Size: outBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer oBuf.Destroy()

	causal := uint32(0)
	if cfg.Causal {
		causal = 1
	}
	p := mhaParams{
		NumHeads: uint32(cfg.NumHeads), NumKVHeads: uint32(cfg.NumKVHeads),
		HeadDim: uint32(cfg.HeadDim), SeqLen: uint32(cfg.SeqLen),
		KVOffset: uint32(cfg.KVOffset), MaxSeqLen: uint32(cfg.MaxSeqLen),
		TileSize: uint32(cfg.TileSize), Causal: causal, KvLen: uint32(cfg.KvLen),
	}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-mha-p", Contents: wgpu.ToBytes([]mhaParams{p}),
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
			{Binding: 1, Buffer: qBuf, Offset: 0, Size: qBuf.GetSize()},
			{Binding: 2, Buffer: kBuf, Offset: 0, Size: kBuf.GetSize()},
			{Binding: 3, Buffer: vBuf, Offset: 0, Size: vBuf.GetSize()},
			{Binding: 4, Buffer: oBuf, Offset: 0, Size: oBuf.GetSize()},
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
	pass.DispatchWorkgroups(uint32(cfg.NumHeads), uint32(cfg.SeqLen), 1)
	pass.End()
	cmd, err := enc.Finish(nil)
	if err != nil {
		return err
	}
	qDev.Submit(cmd)

	outData, err := readbackF32(dev, qDev, oBuf, outN)
	if err != nil {
		return err
	}
	copy(out, outData)
	return nil
}

// ShaderRoPE — in-place rotary positional encoding (ported from loom/poly).
const ShaderRoPE = `
struct RoPEParams {
    seqLen: u32,
    headDim: u32,
    numHeads: u32,
    offset: u32,
    theta: f32,
};

@group(0) @binding(0) var<uniform> params: RoPEParams;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    let halfDim = params.headDim / 2u;

    let totalPairs = params.seqLen * params.numHeads * halfDim;
    if (tid >= totalPairs) { return; }

    let d = tid % halfDim;
    let h = (tid / halfDim) % params.numHeads;
    let s = tid / (halfDim * params.numHeads);

    let pos = f32(params.offset + s);
    let freq = 1.0 / pow(params.theta, f32(2u * d) / f32(params.headDim));
    let angle = pos * freq;

    let cos_val = cos(angle);
    let sin_val = sin(angle);

    let idx0 = (s * params.numHeads + h) * params.headDim + d;
    let idx1 = idx0 + halfDim;

    let v0 = data[idx0];
    let v1 = data[idx1];

    data[idx0] = v0 * cos_val - v1 * sin_val;
    data[idx1] = v0 * sin_val + v1 * cos_val;
}
`

// ShaderKVUpdate writes new K/V rows into [numKVHeads, maxSeqLen, headDim] caches.
const ShaderKVUpdate = `
struct KVParams {
    offset: u32,
    headDim: u32,
    maxSeqLen: u32,
    numKVHeads: u32,
    numTokens: u32,
    pad0: u32, pad1: u32, pad2: u32, pad3: u32, pad4: u32, pad5: u32, pad6: u32,
};
@group(0) @binding(0) var<storage, read_write> kCache: array<f32>;
@group(0) @binding(1) var<storage, read_write> vCache: array<f32>;
@group(0) @binding(2) var<storage, read> newK: array<f32>;
@group(0) @binding(3) var<storage, read> newV: array<f32>;
@group(0) @binding(4) var<uniform> params: KVParams;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    let kvDim = params.numKVHeads * params.headDim;
    if (tid >= kvDim * params.numTokens) { return; }

    let tokenIdx = tid / kvDim;
    let dimIdx = tid % kvDim;
    let h = dimIdx / params.headDim;
    let d = dimIdx % params.headDim;

    let cacheIdx = (h * params.maxSeqLen + params.offset + tokenIdx) * params.headDim + d;
    kCache[cacheIdx] = newK[tid];
    vCache[cacheIdx] = newV[tid];
}
`
