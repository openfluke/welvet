package webgpu

import (
	"fmt"
	"math"

	"github.com/openfluke/webgpu/wgpu"
)

type mhaBwdParams struct {
	NumHeads   uint32
	NumKVHeads uint32
	HeadDim    uint32
	SeqLen     uint32
	KVOffset   uint32
	MaxSeqLen  uint32
	KvLen      uint32
	Causal     uint32
	Scale      float32
	_          [3]uint32
}

// MHABackward computes dQ / dK / dV on device for SoftmaxStandard attention.
//
// Layouts match MHAForward:
//   - gradOut, Q, dQ: [seqLen, numHeads, headDim]
//   - kCache, vCache, dK, dV: [numKVHeads, maxSeqLen, headDim]
//
// One workgroup per head runs the full query loop serially so dK/dV have no races.
// Causal: query s (abs pos KVOffset+s) attends to keys 0..KVOffset+s.
// Bidirectional: every query attends to keys 0..KvLen-1.
func MHABackward(gradOut, q, kCache, vCache, dQ, dK, dV []float32, cfg MHAConfig) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu MHABackward: %w", initErr)
	}
	if cfg.MaxSeqLen <= 0 || cfg.MaxSeqLen > 2048 {
		return fmt.Errorf("webgpu MHABackward: MaxSeqLen must be in 1..2048 (got %d)", cfg.MaxSeqLen)
	}
	qN := cfg.SeqLen * cfg.NumHeads * cfg.HeadDim
	cacheN := cfg.NumKVHeads * cfg.MaxSeqLen * cfg.HeadDim
	if len(gradOut) < qN || len(q) < qN || len(dQ) < qN ||
		len(kCache) < cacheN || len(vCache) < cacheN || len(dK) < cacheN || len(dV) < cacheN {
		return fmt.Errorf("webgpu MHABackward: shape")
	}
	if cfg.NumHeads <= 0 || cfg.NumKVHeads <= 0 || cfg.HeadDim <= 0 || cfg.NumHeads%cfg.NumKVHeads != 0 {
		return fmt.Errorf("webgpu MHABackward: head geometry")
	}
	if !cfg.Causal && cfg.KvLen <= 0 {
		return fmt.Errorf("webgpu MHABackward: bidirectional needs KvLen>0")
	}
	if err := sess.ensureMHABwdPipe(); err != nil {
		return err
	}
	scale := float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
	return sess.mhaBackward(gradOut[:qN], q[:qN], kCache[:cacheN], vCache[:cacheN],
		dQ[:qN], dK[:cacheN], dV[:cacheN], cfg, scale)
}

func (s *session) ensureMHABwdPipe() error {
	if s.pipeMHABwd != nil {
		return nil
	}
	var err error
	s.pipeMHABwd, err = makePipeline(s.device, ShaderMHABackward, "welvet-mha-bwd")
	return err
}

func (s *session) mhaBackward(gradOut, q, kCache, vCache, dQ, dK, dV []float32, cfg MHAConfig, scale float32) error {
	dev, qdev := s.device, s.queue
	mkRO := func(label string, data []float32) (*wgpu.Buffer, error) {
		return dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
			Label: label, Contents: wgpu.ToBytes(data),
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
		})
	}
	mkRW := func(label string, n int) (*wgpu.Buffer, error) {
		bytes := uint64(n * 4)
		if bytes < 64 {
			bytes = 64
		}
		zeros := make([]float32, n)
		return dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
			Label: label, Contents: wgpu.ToBytes(zeros),
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
		})
	}

	goBuf, err := mkRO("welvet-mhab-go", gradOut)
	if err != nil {
		return err
	}
	defer goBuf.Destroy()
	qBuf, err := mkRO("welvet-mhab-q", q)
	if err != nil {
		return err
	}
	defer qBuf.Destroy()
	kBuf, err := mkRO("welvet-mhab-k", kCache)
	if err != nil {
		return err
	}
	defer kBuf.Destroy()
	vBuf, err := mkRO("welvet-mhab-v", vCache)
	if err != nil {
		return err
	}
	defer vBuf.Destroy()

	dQBuf, err := mkRW("welvet-mhab-dq", len(dQ))
	if err != nil {
		return err
	}
	defer dQBuf.Destroy()
	dKBuf, err := mkRW("welvet-mhab-dk", len(dK))
	if err != nil {
		return err
	}
	defer dKBuf.Destroy()
	dVBuf, err := mkRW("welvet-mhab-dv", len(dV))
	if err != nil {
		return err
	}
	defer dVBuf.Destroy()

	causal := uint32(0)
	if cfg.Causal {
		causal = 1
	}
	p := mhaBwdParams{
		NumHeads: uint32(cfg.NumHeads), NumKVHeads: uint32(cfg.NumKVHeads),
		HeadDim: uint32(cfg.HeadDim), SeqLen: uint32(cfg.SeqLen),
		KVOffset: uint32(cfg.KVOffset), MaxSeqLen: uint32(cfg.MaxSeqLen),
		KvLen: uint32(cfg.KvLen), Causal: causal, Scale: scale,
	}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-mhab-p", Contents: wgpu.ToBytes([]mhaBwdParams{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()

	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.pipeMHABwd.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: goBuf, Offset: 0, Size: goBuf.GetSize()},
			{Binding: 2, Buffer: qBuf, Offset: 0, Size: qBuf.GetSize()},
			{Binding: 3, Buffer: kBuf, Offset: 0, Size: kBuf.GetSize()},
			{Binding: 4, Buffer: vBuf, Offset: 0, Size: vBuf.GetSize()},
			{Binding: 5, Buffer: dQBuf, Offset: 0, Size: dQBuf.GetSize()},
			{Binding: 6, Buffer: dKBuf, Offset: 0, Size: dKBuf.GetSize()},
			{Binding: 7, Buffer: dVBuf, Offset: 0, Size: dVBuf.GetSize()},
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
	pass.SetPipeline(s.pipeMHABwd)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(uint32(cfg.NumHeads), 1, 1)
	pass.End()
	cmd, err := enc.Finish(nil)
	if err != nil {
		return err
	}
	qdev.Submit(cmd)

	outDQ, err := readbackF32(dev, qdev, dQBuf, len(dQ))
	if err != nil {
		return err
	}
	copy(dQ, outDQ)
	outDK, err := readbackF32(dev, qdev, dKBuf, len(dK))
	if err != nil {
		return err
	}
	copy(dK, outDK)
	outDV, err := readbackF32(dev, qdev, dVBuf, len(dV))
	if err != nil {
		return err
	}
	copy(dV, outDV)
	return nil
}

// ShaderMHABackward — one workgroup per head; serial over queries (race-free dK/dV).
// Layouts match MHAForward. Max key length 2048.
const ShaderMHABackward = `
struct Params {
    numHeads: u32,
    numKVHeads: u32,
    headDim: u32,
    seqLen: u32,
    kvOffset: u32,
    maxSeqLen: u32,
    kvLen: u32,
    causal: u32,
    scale: f32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gradOut: array<f32>;
@group(0) @binding(2) var<storage, read> Q: array<f32>;
@group(0) @binding(3) var<storage, read> K: array<f32>;
@group(0) @binding(4) var<storage, read> V: array<f32>;
@group(0) @binding(5) var<storage, read_write> dQ: array<f32>;
@group(0) @binding(6) var<storage, read_write> dK: array<f32>;
@group(0) @binding(7) var<storage, read_write> dV: array<f32>;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x;
    if (h >= params.numHeads) { return; }
    let headDim = params.headDim;
    let kvGroup = params.numHeads / params.numKVHeads;
    let kvH = h / kvGroup;
    let seqLen = params.seqLen;

    for (var s_q: u32 = 0u; s_q < seqLen; s_q++) {
        let absQ = params.kvOffset + s_q;
        var kEnd: u32 = params.kvLen;
        if (params.causal != 0u) {
            kEnd = absQ + 1u;
        }
        if (kEnd > params.maxSeqLen) { kEnd = params.maxSeqLen; }
        if (kEnd > 2048u) { return; }

        var scores: array<f32, 2048>;
        var max_score: f32 = -1e38;
        let qBase = (s_q * params.numHeads + h) * headDim;

        for (var s_k: u32 = 0u; s_k < kEnd; s_k++) {
            var dot: f32 = 0.0;
            let kBase = (kvH * params.maxSeqLen + s_k) * headDim;
            for (var d: u32 = 0u; d < headDim; d++) {
                dot += Q[qBase + d] * K[kBase + d];
            }
            let score = dot * params.scale;
            scores[s_k] = score;
            if (score > max_score) { max_score = score; }
        }

        var exp_sum: f32 = 0.0;
        for (var s_k: u32 = 0u; s_k < kEnd; s_k++) {
            scores[s_k] = exp(scores[s_k] - max_score);
            exp_sum += scores[s_k];
        }
        let inv = 1.0 / exp_sum;
        for (var s_k: u32 = 0u; s_k < kEnd; s_k++) {
            scores[s_k] *= inv;
        }

        var dW: array<f32, 2048>;
        var sum_dw_w: f32 = 0.0;
        for (var s_k: u32 = 0u; s_k < kEnd; s_k++) {
            var ds: f32 = 0.0;
            let vBase = (kvH * params.maxSeqLen + s_k) * headDim;
            for (var d: u32 = 0u; d < headDim; d++) {
                let go = gradOut[qBase + d];
                dV[vBase + d] += scores[s_k] * go;
                ds += go * V[vBase + d];
            }
            dW[s_k] = ds;
            sum_dw_w += ds * scores[s_k];
        }

        for (var s_k: u32 = 0u; s_k < kEnd; s_k++) {
            let d_logit = scores[s_k] * (dW[s_k] - sum_dw_w) * params.scale;
            let kBase = (kvH * params.maxSeqLen + s_k) * headDim;
            for (var d: u32 = 0u; d < headDim; d++) {
                dQ[qBase + d] += d_logit * K[kBase + d];
                dK[kBase + d] += d_logit * Q[qBase + d];
            }
        }
    }
}
`
