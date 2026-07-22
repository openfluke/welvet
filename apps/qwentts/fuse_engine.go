package qwentts

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/openfluke/webgpu/wgpu"
	"github.com/openfluke/welvet/tiling"
	"github.com/openfluke/welvet/webgpu"
)

// talkerFuse runs one Qwen3 talker decode token as a single WebGPU submit
// ("true" fuse): RMSNorm -> Q/K/V GEMV -> head-RMS -> RoPE -> WriteKV ->
// GQA attn -> O GEMV -> residual -> RMSNorm -> gate/up GEMV -> SwiGLU ->
// down GEMV -> residual, for every layer, then a final RMSNorm; one readback.
//
// All per-layer weights and the KV cache stay resident in VRAM. Position is a
// GPU-resident counter (stepBuf) advanced on device each token.
type talkerFuse struct {
	H, hd, nh, nkv, inter, maxSeq int
	qDim, kvDim                   int
	nLayers                       int
	ropeTheta                     float32
	epsBits                       uint32

	pipeGEMV, pipeRMS, pipeHeadRMS, pipeRes, pipeRoPE, pipeWKV, pipeAttn, pipeSiLU, pipeInc *wgpu.ComputePipeline

	// scratch (per token, reused across layers)
	stepBuf   *wgpu.Buffer
	xBuf      *wgpu.Buffer // residual stream [H]
	xnBuf     *wgpu.Buffer // normed [H]
	qBuf      *wgpu.Buffer // [qDim]
	kBuf      *wgpu.Buffer // [kvDim]
	vBuf      *wgpu.Buffer // [kvDim]
	attnBuf   *wgpu.Buffer // [qDim]
	projBuf   *wgpu.Buffer // [H]
	gateBuf   *wgpu.Buffer // [inter]
	upBuf     *wgpu.Buffer // [inter]
	interBuf  *wgpu.Buffer // [inter]
	finalNorm *wgpu.Buffer // [H]
	dummyBias *wgpu.Buffer // [1]

	layers  []fuseTalkerLayerGPU
	finalLN *wgpu.Buffer // t.norm [H]
}

type fuseTalkerLayerGPU struct {
	wq, wk, wv, wo    *wgpu.Buffer
	wgate, wup, wdown *wgpu.Buffer
	inputLN, postLN   *wgpu.Buffer
	qNorm, kNorm      *wgpu.Buffer
	kCache, vCache    *wgpu.Buffer
}

// fuseCtx keeps ephemeral uniform buffers / bind groups alive until Submit.
type fuseCtx struct {
	uniforms []*wgpu.Buffer
	bgs      []*wgpu.BindGroup
}

func (c *fuseCtx) release() {
	for _, bg := range c.bgs {
		if bg != nil {
			bg.Release()
		}
	}
	for _, u := range c.uniforms {
		if u != nil {
			u.Destroy()
		}
	}
	c.bgs = nil
	c.uniforms = nil
}

// newTalkerFuse uploads all talker layer weights + KV caches to VRAM and
// compiles the decode pipelines. maxSeq is clamped to [1,2048].
func newTalkerFuse(t *Talker, maxSeq int) (*talkerFuse, error) {
	if t == nil || len(t.layers) == 0 {
		return nil, fmt.Errorf("talkerFuse: empty talker")
	}
	cfg := t.cfg
	if maxSeq <= 0 {
		maxSeq = 2048
	}
	if maxSeq > 2048 {
		maxSeq = 2048
	}
	if cfg.HeadDim > 128 {
		return nil, fmt.Errorf("talkerFuse: headDim %d exceeds 128", cfg.HeadDim)
	}
	e := &talkerFuse{
		H:         cfg.HiddenSize,
		hd:        cfg.HeadDim,
		nh:        cfg.NumHeads,
		nkv:       cfg.NumKVHeads,
		inter:     cfg.IntermediateSize,
		maxSeq:    maxSeq,
		qDim:      cfg.NumHeads * cfg.HeadDim,
		kvDim:     cfg.NumKVHeads * cfg.HeadDim,
		nLayers:   cfg.NumLayers,
		ropeTheta: float32(cfg.RopeTheta),
		epsBits:   math.Float32bits(float32(cfg.RMSNormEps)),
	}
	if e.H == 0 || e.hd == 0 || e.nh == 0 || e.nkv == 0 {
		return nil, fmt.Errorf("talkerFuse: bad dims")
	}
	err := webgpu.WithDevice(func(dev *wgpu.Device, _ *wgpu.Queue) error {
		return e.initLocked(dev, t)
	})
	if err != nil {
		e.Close()
		return nil, err
	}
	return e, nil
}

func (e *talkerFuse) initLocked(dev *wgpu.Device, t *Talker) error {
	var err error
	mk := func(code, label string) (*wgpu.ComputePipeline, error) {
		return webgpu.MakeComputePipeline(dev, code, label)
	}
	if e.pipeGEMV, err = mk(shaderQwenGEMV, "qwen-fuse-gemv"); err != nil {
		return err
	}
	if e.pipeRMS, err = mk(shaderQwenRMSNorm, "qwen-fuse-rms"); err != nil {
		return err
	}
	if e.pipeHeadRMS, err = mk(shaderQwenHeadRMS, "qwen-fuse-headrms"); err != nil {
		return err
	}
	if e.pipeRes, err = mk(shaderQwenResidual, "qwen-fuse-res"); err != nil {
		return err
	}
	if e.pipeRoPE, err = mk(shaderQwenRoPE, "qwen-fuse-rope"); err != nil {
		return err
	}
	if e.pipeWKV, err = mk(shaderQwenWriteKV, "qwen-fuse-wkv"); err != nil {
		return err
	}
	if e.pipeAttn, err = mk(shaderQwenAttn, "qwen-fuse-attn"); err != nil {
		return err
	}
	if e.pipeSiLU, err = mk(shaderQwenSiLUMul, "qwen-fuse-silu"); err != nil {
		return err
	}
	if e.pipeInc, err = mk(shaderQwenIncPos, "qwen-fuse-inc"); err != nil {
		return err
	}

	mkStor := func(label string, n int) (*wgpu.Buffer, error) {
		sz := uint64(n * 4)
		if sz < 64 {
			sz = 64
		}
		return dev.CreateBuffer(&wgpu.BufferDescriptor{
			Label: label, Size: sz,
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
		})
	}
	mkInit := func(label string, data []float32) (*wgpu.Buffer, error) {
		if len(data) == 0 {
			return mkStor(label, 1)
		}
		return dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
			Label: label, Contents: wgpu.ToBytes(data),
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
		})
	}

	if e.stepBuf, err = dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "qwen-step", Contents: packU32(0),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	}); err != nil {
		return err
	}
	if e.xBuf, err = mkStor("qwen-x", e.H); err != nil {
		return err
	}
	if e.xnBuf, err = mkStor("qwen-xn", e.H); err != nil {
		return err
	}
	if e.qBuf, err = mkStor("qwen-q", e.qDim); err != nil {
		return err
	}
	if e.kBuf, err = mkStor("qwen-k", e.kvDim); err != nil {
		return err
	}
	if e.vBuf, err = mkStor("qwen-v", e.kvDim); err != nil {
		return err
	}
	if e.attnBuf, err = mkStor("qwen-attn", e.qDim); err != nil {
		return err
	}
	if e.projBuf, err = mkStor("qwen-proj", e.H); err != nil {
		return err
	}
	if e.gateBuf, err = mkStor("qwen-gate", e.inter); err != nil {
		return err
	}
	if e.upBuf, err = mkStor("qwen-up", e.inter); err != nil {
		return err
	}
	if e.interBuf, err = mkStor("qwen-inter", e.inter); err != nil {
		return err
	}
	if e.finalNorm, err = mkStor("qwen-fnorm", e.H); err != nil {
		return err
	}
	if e.dummyBias, err = mkStor("qwen-dummy-b", 1); err != nil {
		return err
	}
	if e.finalLN, err = mkInit("qwen-norm", t.norm); err != nil {
		return err
	}

	kvElems := e.nkv * e.maxSeq * e.hd
	e.layers = make([]fuseTalkerLayerGPU, e.nLayers)
	for i := range t.layers {
		l := &t.layers[i]
		L := &e.layers[i]
		if L.wq, err = mkInit(fmt.Sprintf("wq%d", i), l.q.W); err != nil {
			return err
		}
		if L.wk, err = mkInit(fmt.Sprintf("wk%d", i), l.k.W); err != nil {
			return err
		}
		if L.wv, err = mkInit(fmt.Sprintf("wv%d", i), l.v.W); err != nil {
			return err
		}
		if L.wo, err = mkInit(fmt.Sprintf("wo%d", i), l.o.W); err != nil {
			return err
		}
		if L.wgate, err = mkInit(fmt.Sprintf("wg%d", i), l.gate.W); err != nil {
			return err
		}
		if L.wup, err = mkInit(fmt.Sprintf("wu%d", i), l.up.W); err != nil {
			return err
		}
		if L.wdown, err = mkInit(fmt.Sprintf("wd%d", i), l.down.W); err != nil {
			return err
		}
		if L.inputLN, err = mkInit(fmt.Sprintf("iln%d", i), l.inputLN); err != nil {
			return err
		}
		if L.postLN, err = mkInit(fmt.Sprintf("pln%d", i), l.postLN); err != nil {
			return err
		}
		if L.qNorm, err = mkInit(fmt.Sprintf("qn%d", i), l.qNorm); err != nil {
			return err
		}
		if L.kNorm, err = mkInit(fmt.Sprintf("kn%d", i), l.kNorm); err != nil {
			return err
		}
		if L.kCache, err = mkStor(fmt.Sprintf("kc%d", i), kvElems); err != nil {
			return err
		}
		if L.vCache, err = mkStor(fmt.Sprintf("vc%d", i), kvElems); err != nil {
			return err
		}
	}
	return nil
}

// Close releases all GPU buffers.
func (e *talkerFuse) Close() {
	if e == nil {
		return
	}
	_ = webgpu.WithDevice(func(dev *wgpu.Device, _ *wgpu.Queue) error {
		d := func(b *wgpu.Buffer) {
			if b != nil {
				b.Destroy()
			}
		}
		d(e.stepBuf)
		d(e.xBuf)
		d(e.xnBuf)
		d(e.qBuf)
		d(e.kBuf)
		d(e.vBuf)
		d(e.attnBuf)
		d(e.projBuf)
		d(e.gateBuf)
		d(e.upBuf)
		d(e.interBuf)
		d(e.finalNorm)
		d(e.dummyBias)
		d(e.finalLN)
		for i := range e.layers {
			L := &e.layers[i]
			d(L.wq)
			d(L.wk)
			d(L.wv)
			d(L.wo)
			d(L.wgate)
			d(L.wup)
			d(L.wdown)
			d(L.inputLN)
			d(L.postLN)
			d(L.qNorm)
			d(L.kNorm)
			d(L.kCache)
			d(L.vCache)
		}
		return nil
	})
	e.layers = nil
}

// ResetPos sets the decode position back to 0 (call before a new prefill).
func (e *talkerFuse) ResetPos() error {
	if e == nil {
		return nil
	}
	return webgpu.WithDevice(func(dev *wgpu.Device, q *wgpu.Queue) error {
		q.WriteBuffer(e.stepBuf, 0, packU32(0))
		return nil
	})
}

// DecodeStepFused writes x[:H] as the residual, runs all layers + final
// RMSNorm in one submit, advances pos on device, and reads back the final
// post-norm hidden [H] into x.
func (e *talkerFuse) DecodeStepFused(x []float32) ([]float32, error) {
	if e == nil {
		return nil, fmt.Errorf("nil fuse")
	}
	if len(x) < e.H {
		return nil, fmt.Errorf("fuse: short x (%d < %d)", len(x), e.H)
	}
	var out []float32
	err := webgpu.WithDevice(func(dev *wgpu.Device, q *wgpu.Queue) error {
		ctx := &fuseCtx{}
		defer ctx.release()
		q.WriteBuffer(e.xBuf, 0, wgpu.ToBytes(x[:e.H]))
		enc, err := dev.CreateCommandEncoder(nil)
		if err != nil {
			return err
		}
		pass := enc.BeginComputePass(nil)
		for i := range e.layers {
			if err := e.recordLayer(dev, pass, ctx, &e.layers[i]); err != nil {
				pass.End()
				return err
			}
		}
		// final RMSNorm -> finalNorm
		if err := e.dispatchRMS(dev, pass, ctx, e.xBuf, e.finalLN, e.finalNorm); err != nil {
			pass.End()
			return err
		}
		if err := e.dispatchInc(dev, pass, ctx); err != nil {
			pass.End()
			return err
		}
		pass.End()
		cmd, err := enc.Finish(nil)
		if err != nil {
			return err
		}
		q.Submit(cmd)
		res, err := webgpu.ReadbackF32(dev, q, e.finalNorm, e.H)
		if err != nil {
			return err
		}
		out = res
		return nil
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (e *talkerFuse) recordLayer(dev *wgpu.Device, pass *wgpu.ComputePassEncoder, ctx *fuseCtx, L *fuseTalkerLayerGPU) error {
	// attn pre-norm
	if err := e.dispatchRMS(dev, pass, ctx, e.xBuf, L.inputLN, e.xnBuf); err != nil {
		return err
	}
	// Q/K/V projections
	if err := e.dispatchGEMV(dev, pass, ctx, e.xnBuf, L.wq, e.qBuf, e.H, e.qDim); err != nil {
		return err
	}
	if err := e.dispatchGEMV(dev, pass, ctx, e.xnBuf, L.wk, e.kBuf, e.H, e.kvDim); err != nil {
		return err
	}
	if err := e.dispatchGEMV(dev, pass, ctx, e.xnBuf, L.wv, e.vBuf, e.H, e.kvDim); err != nil {
		return err
	}
	// per-head q/k norm
	if err := e.dispatchHeadRMS(dev, pass, ctx, e.qBuf, L.qNorm, e.nh); err != nil {
		return err
	}
	if err := e.dispatchHeadRMS(dev, pass, ctx, e.kBuf, L.kNorm, e.nkv); err != nil {
		return err
	}
	// RoPE
	if err := e.dispatchRoPE(dev, pass, ctx, e.qBuf, e.nh); err != nil {
		return err
	}
	if err := e.dispatchRoPE(dev, pass, ctx, e.kBuf, e.nkv); err != nil {
		return err
	}
	// append KV + attention
	if err := e.dispatchWriteKV(dev, pass, ctx, L.kCache, L.vCache); err != nil {
		return err
	}
	if err := e.dispatchAttn(dev, pass, ctx, L.kCache, L.vCache); err != nil {
		return err
	}
	// o_proj + residual
	if err := e.dispatchGEMV(dev, pass, ctx, e.attnBuf, L.wo, e.projBuf, e.qDim, e.H); err != nil {
		return err
	}
	if err := e.dispatchRes(dev, pass, ctx, e.projBuf, e.xBuf, e.H); err != nil {
		return err
	}
	// mlp pre-norm
	if err := e.dispatchRMS(dev, pass, ctx, e.xBuf, L.postLN, e.xnBuf); err != nil {
		return err
	}
	// gate/up + SwiGLU
	if err := e.dispatchGEMV(dev, pass, ctx, e.xnBuf, L.wgate, e.gateBuf, e.H, e.inter); err != nil {
		return err
	}
	if err := e.dispatchGEMV(dev, pass, ctx, e.xnBuf, L.wup, e.upBuf, e.H, e.inter); err != nil {
		return err
	}
	if err := e.dispatchSiLU(dev, pass, ctx); err != nil {
		return err
	}
	// down + residual
	if err := e.dispatchGEMV(dev, pass, ctx, e.interBuf, L.wdown, e.projBuf, e.inter, e.H); err != nil {
		return err
	}
	return e.dispatchRes(dev, pass, ctx, e.projBuf, e.xBuf, e.H)
}

// --- dispatch helpers -------------------------------------------------------

func packU32(v uint32) []byte {
	b := make([]byte, 4)
	binary.LittleEndian.PutUint32(b, v)
	return b
}

func packUniform(words ...uint32) []byte {
	b := make([]byte, len(words)*4)
	for i, v := range words {
		binary.LittleEndian.PutUint32(b[i*4:], v)
	}
	return b
}

func (e *talkerFuse) mkUniform(dev *wgpu.Device, ctx *fuseCtx, words ...uint32) (*wgpu.Buffer, error) {
	u, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "qwen-u", Contents: packUniform(words...),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, err
	}
	ctx.uniforms = append(ctx.uniforms, u)
	return u, nil
}

func (e *talkerFuse) bind(dev *wgpu.Device, ctx *fuseCtx, pipe *wgpu.ComputePipeline, entries []wgpu.BindGroupEntry) (*wgpu.BindGroup, error) {
	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout:  pipe.GetBindGroupLayout(0),
		Entries: entries,
	})
	if err != nil {
		return nil, err
	}
	ctx.bgs = append(ctx.bgs, bg)
	return bg, nil
}

func (e *talkerFuse) dispatchGEMV(dev *wgpu.Device, pass *wgpu.ComputePassEncoder, ctx *fuseCtx, x, w, y *wgpu.Buffer, in, out int) error {
	u, err := e.mkUniform(dev, ctx, uint32(in), uint32(out), 0, 0)
	if err != nil {
		return err
	}
	bg, err := e.bind(dev, ctx, e.pipeGEMV, []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: u, Size: u.GetSize()},
		{Binding: 1, Buffer: x, Size: x.GetSize()},
		{Binding: 2, Buffer: w, Size: w.GetSize()},
		{Binding: 3, Buffer: e.dummyBias, Size: e.dummyBias.GetSize()},
		{Binding: 4, Buffer: y, Size: y.GetSize()},
	})
	if err != nil {
		return err
	}
	pass.SetPipeline(e.pipeGEMV)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(tiling.GPUWorkgroupsX(out, 64), 1, 1)
	return nil
}

func (e *talkerFuse) dispatchRMS(dev *wgpu.Device, pass *wgpu.ComputePassEncoder, ctx *fuseCtx, in, w, out *wgpu.Buffer) error {
	u, err := e.mkUniform(dev, ctx, uint32(e.H), e.epsBits, 0, 0)
	if err != nil {
		return err
	}
	bg, err := e.bind(dev, ctx, e.pipeRMS, []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: u, Size: u.GetSize()},
		{Binding: 1, Buffer: in, Size: in.GetSize()},
		{Binding: 2, Buffer: w, Size: w.GetSize()},
		{Binding: 3, Buffer: out, Size: out.GetSize()},
	})
	if err != nil {
		return err
	}
	pass.SetPipeline(e.pipeRMS)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(1, 1, 1)
	return nil
}

func (e *talkerFuse) dispatchHeadRMS(dev *wgpu.Device, pass *wgpu.ComputePassEncoder, ctx *fuseCtx, x, gamma *wgpu.Buffer, heads int) error {
	u, err := e.mkUniform(dev, ctx, uint32(heads), uint32(e.hd), e.epsBits, 0)
	if err != nil {
		return err
	}
	bg, err := e.bind(dev, ctx, e.pipeHeadRMS, []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: u, Size: u.GetSize()},
		{Binding: 1, Buffer: x, Size: x.GetSize()},
		{Binding: 2, Buffer: gamma, Size: gamma.GetSize()},
	})
	if err != nil {
		return err
	}
	pass.SetPipeline(e.pipeHeadRMS)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(uint32(heads), 1, 1)
	return nil
}

func (e *talkerFuse) dispatchRes(dev *wgpu.Device, pass *wgpu.ComputePassEncoder, ctx *fuseCtx, addend, inout *wgpu.Buffer, n int) error {
	u, err := e.mkUniform(dev, ctx, uint32(n), 0, 0, 0)
	if err != nil {
		return err
	}
	bg, err := e.bind(dev, ctx, e.pipeRes, []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: u, Size: u.GetSize()},
		{Binding: 1, Buffer: addend, Size: addend.GetSize()},
		{Binding: 2, Buffer: inout, Size: inout.GetSize()},
	})
	if err != nil {
		return err
	}
	pass.SetPipeline(e.pipeRes)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(tiling.GPUWorkgroupsX(n, 64), 1, 1)
	return nil
}

func (e *talkerFuse) dispatchRoPE(dev *wgpu.Device, pass *wgpu.ComputePassEncoder, ctx *fuseCtx, x *wgpu.Buffer, heads int) error {
	u, err := e.mkUniform(dev, ctx, uint32(heads), uint32(e.hd), math.Float32bits(e.ropeTheta), 0)
	if err != nil {
		return err
	}
	bg, err := e.bind(dev, ctx, e.pipeRoPE, []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: u, Size: u.GetSize()},
		{Binding: 1, Buffer: e.stepBuf, Size: e.stepBuf.GetSize()},
		{Binding: 2, Buffer: x, Size: x.GetSize()},
	})
	if err != nil {
		return err
	}
	pass.SetPipeline(e.pipeRoPE)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(tiling.GPUWorkgroupsX(heads, 64), 1, 1)
	return nil
}

func (e *talkerFuse) dispatchWriteKV(dev *wgpu.Device, pass *wgpu.ComputePassEncoder, ctx *fuseCtx, kCache, vCache *wgpu.Buffer) error {
	u, err := e.mkUniform(dev, ctx, uint32(e.kvDim), uint32(e.maxSeq), uint32(e.hd), 0)
	if err != nil {
		return err
	}
	bg, err := e.bind(dev, ctx, e.pipeWKV, []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: u, Size: u.GetSize()},
		{Binding: 1, Buffer: e.stepBuf, Size: e.stepBuf.GetSize()},
		{Binding: 2, Buffer: e.kBuf, Size: e.kBuf.GetSize()},
		{Binding: 3, Buffer: e.vBuf, Size: e.vBuf.GetSize()},
		{Binding: 4, Buffer: kCache, Size: kCache.GetSize()},
		{Binding: 5, Buffer: vCache, Size: vCache.GetSize()},
	})
	if err != nil {
		return err
	}
	pass.SetPipeline(e.pipeWKV)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(tiling.GPUWorkgroupsX(e.kvDim, 64), 1, 1)
	return nil
}

func (e *talkerFuse) dispatchAttn(dev *wgpu.Device, pass *wgpu.ComputePassEncoder, ctx *fuseCtx, kCache, vCache *wgpu.Buffer) error {
	u, err := e.mkUniform(dev, ctx, uint32(e.nh), uint32(e.nkv), uint32(e.hd), uint32(e.maxSeq))
	if err != nil {
		return err
	}
	bg, err := e.bind(dev, ctx, e.pipeAttn, []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: u, Size: u.GetSize()},
		{Binding: 1, Buffer: e.stepBuf, Size: e.stepBuf.GetSize()},
		{Binding: 2, Buffer: e.qBuf, Size: e.qBuf.GetSize()},
		{Binding: 3, Buffer: kCache, Size: kCache.GetSize()},
		{Binding: 4, Buffer: vCache, Size: vCache.GetSize()},
		{Binding: 5, Buffer: e.attnBuf, Size: e.attnBuf.GetSize()},
	})
	if err != nil {
		return err
	}
	pass.SetPipeline(e.pipeAttn)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(uint32(e.nh), 1, 1)
	return nil
}

func (e *talkerFuse) dispatchSiLU(dev *wgpu.Device, pass *wgpu.ComputePassEncoder, ctx *fuseCtx) error {
	u, err := e.mkUniform(dev, ctx, uint32(e.inter), 0, 0, 0)
	if err != nil {
		return err
	}
	bg, err := e.bind(dev, ctx, e.pipeSiLU, []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: u, Size: u.GetSize()},
		{Binding: 1, Buffer: e.gateBuf, Size: e.gateBuf.GetSize()},
		{Binding: 2, Buffer: e.upBuf, Size: e.upBuf.GetSize()},
		{Binding: 3, Buffer: e.interBuf, Size: e.interBuf.GetSize()},
	})
	if err != nil {
		return err
	}
	pass.SetPipeline(e.pipeSiLU)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(tiling.GPUWorkgroupsX(e.inter, 64), 1, 1)
	return nil
}

func (e *talkerFuse) dispatchInc(dev *wgpu.Device, pass *wgpu.ComputePassEncoder, ctx *fuseCtx) error {
	bg, err := e.bind(dev, ctx, e.pipeInc, []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: e.stepBuf, Size: e.stepBuf.GetSize()},
	})
	if err != nil {
		return err
	}
	pass.SetPipeline(e.pipeInc)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(1, 1, 1)
	return nil
}
