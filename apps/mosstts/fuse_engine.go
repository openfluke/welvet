package mosstts

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/openfluke/webgpu/wgpu"
	"github.com/openfluke/welvet/tiling"
	"github.com/openfluke/welvet/webgpu"
)

// gpt2Fuse runs one GPT-2 decode token as a single WebGPU submit (true fuse).
type gpt2Fuse struct {
	hidden, heads, headDim, ff, maxSeq int
	nLayers                            int
	ropeBase                           float32
	scaleAttn                          bool
	epsBits                            uint32

	pipeGEMV, pipeLN, pipeRes, pipeGELU, pipeRoPE, pipeWKV, pipeAttn, pipeInc, pipeCopy *wgpu.ComputePipeline

	stepBuf *wgpu.Buffer
	xBuf    *wgpu.Buffer
	normBuf *wgpu.Buffer
	qkvBuf  *wgpu.Buffer
	qBuf    *wgpu.Buffer
	kBuf    *wgpu.Buffer
	vBuf    *wgpu.Buffer
	attnBuf *wgpu.Buffer
	projBuf *wgpu.Buffer
	midBuf  *wgpu.Buffer

	layers    []fuseLayerGPU
	lnfW, lnfB *wgpu.Buffer
	dummyBias *wgpu.Buffer
}

type fuseLayerGPU struct {
	ln1W, ln1B, ln2W, ln2B *wgpu.Buffer
	cAttnW, cAttnB         *wgpu.Buffer
	cProjW, cProjB         *wgpu.Buffer
	fcInW, fcInB           *wgpu.Buffer
	fcOutW, fcOutB         *wgpu.Buffer
	kCache, vCache         *wgpu.Buffer
	hasCAttnB, hasCProjB   bool
	hasFcInB, hasFcOutB    bool
}

// fuseStepCtx keeps ephemeral uniforms/bindgroups alive until after Submit.
type fuseStepCtx struct {
	uniforms []*wgpu.Buffer
	bgs      []*wgpu.BindGroup
}

func (c *fuseStepCtx) release() {
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

func newGPT2Fuse(m *GPT2Model, maxSeq int, ropeBase float64, scaleAttn bool) (*gpt2Fuse, error) {
	if m == nil || len(m.Blocks) == 0 {
		return nil, fmt.Errorf("gpt2Fuse: empty model")
	}
	if maxSeq <= 0 {
		maxSeq = 512
	}
	if maxSeq > 2048 {
		maxSeq = 2048
	}
	b0 := m.Blocks[0]
	heads := b0.Attn.NumHeads
	headDim := b0.Attn.HeadDim
	ff := b0.FcIn.Out
	if heads == 0 || headDim == 0 || m.Hidden == 0 {
		return nil, fmt.Errorf("gpt2Fuse: bad dims")
	}
	if ropeBase <= 0 {
		ropeBase = 10000
	}
	e := &gpt2Fuse{
		hidden: m.Hidden, heads: heads, headDim: headDim, ff: ff, maxSeq: maxSeq,
		nLayers: len(m.Blocks), ropeBase: float32(ropeBase), scaleAttn: scaleAttn,
		epsBits: math.Float32bits(float32(m.Eps)),
	}
	err := webgpu.WithDevice(func(dev *wgpu.Device, q *wgpu.Queue) error {
		return e.initLocked(dev, q, m)
	})
	if err != nil {
		e.Close()
		return nil, err
	}
	return e, nil
}

func (e *gpt2Fuse) initLocked(dev *wgpu.Device, _ *wgpu.Queue, m *GPT2Model) error {
	var err error
	mkPipe := func(code, label string) (*wgpu.ComputePipeline, error) {
		return webgpu.MakeComputePipeline(dev, code, label)
	}
	if e.pipeGEMV, err = mkPipe(shaderFuseGEMVBias, "moss-fuse-gemv"); err != nil {
		return err
	}
	if e.pipeLN, err = mkPipe(shaderFuseLayerNorm, "moss-fuse-ln"); err != nil {
		return err
	}
	if e.pipeRes, err = mkPipe(shaderFuseResidual, "moss-fuse-res"); err != nil {
		return err
	}
	if e.pipeGELU, err = mkPipe(shaderFuseGELU, "moss-fuse-gelu"); err != nil {
		return err
	}
	if e.pipeRoPE, err = mkPipe(shaderFuseRoPE, "moss-fuse-rope"); err != nil {
		return err
	}
	if e.pipeWKV, err = mkPipe(shaderFuseWriteKV, "moss-fuse-wkv"); err != nil {
		return err
	}
	if e.pipeAttn, err = mkPipe(shaderFuseAttnDecode, "moss-fuse-attn"); err != nil {
		return err
	}
	if e.pipeInc, err = mkPipe(shaderFuseIncPos, "moss-fuse-inc"); err != nil {
		return err
	}
	if e.pipeCopy, err = mkPipe(shaderFuseCopySlice, "moss-fuse-copy"); err != nil {
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
		return dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
			Label: label, Contents: wgpu.ToBytes(data),
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
		})
	}
	mkU32 := func(label string, v uint32) (*wgpu.Buffer, error) {
		return dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
			Label: label, Contents: packU32(v),
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
		})
	}

	h, ff := e.hidden, e.ff
	if e.stepBuf, err = mkU32("moss-step", 0); err != nil {
		return err
	}
	if e.xBuf, err = mkStor("moss-x", h); err != nil {
		return err
	}
	if e.normBuf, err = mkStor("moss-norm", h); err != nil {
		return err
	}
	if e.qkvBuf, err = mkStor("moss-qkv", 3*h); err != nil {
		return err
	}
	if e.qBuf, err = mkStor("moss-q", h); err != nil {
		return err
	}
	if e.kBuf, err = mkStor("moss-k", h); err != nil {
		return err
	}
	if e.vBuf, err = mkStor("moss-v", h); err != nil {
		return err
	}
	if e.attnBuf, err = mkStor("moss-attn", h); err != nil {
		return err
	}
	if e.projBuf, err = mkStor("moss-proj", h); err != nil {
		return err
	}
	if e.midBuf, err = mkStor("moss-mid", ff); err != nil {
		return err
	}
	if e.dummyBias, err = mkStor("moss-dummy-b", 1); err != nil {
		return err
	}
	if e.lnfW, err = mkInit("moss-lnf-w", m.LNFW); err != nil {
		return err
	}
	if e.lnfB, err = mkInit("moss-lnf-b", m.LNFB); err != nil {
		return err
	}

	kvElems := e.heads * e.maxSeq * e.headDim
	e.layers = make([]fuseLayerGPU, len(m.Blocks))
	for i := range m.Blocks {
		b := &m.Blocks[i]
		L := &e.layers[i]
		if L.ln1W, err = mkInit(fmt.Sprintf("ln1w%d", i), b.LN1W); err != nil {
			return err
		}
		if L.ln1B, err = mkInit(fmt.Sprintf("ln1b%d", i), b.LN1B); err != nil {
			return err
		}
		if L.ln2W, err = mkInit(fmt.Sprintf("ln2w%d", i), b.LN2W); err != nil {
			return err
		}
		if L.ln2B, err = mkInit(fmt.Sprintf("ln2b%d", i), b.LN2B); err != nil {
			return err
		}
		if L.cAttnW, err = mkInit(fmt.Sprintf("cattnW%d", i), b.Attn.CAttn.W); err != nil {
			return err
		}
		L.hasCAttnB = len(b.Attn.CAttn.B) >= 3*h
		if L.hasCAttnB {
			if L.cAttnB, err = mkInit(fmt.Sprintf("cattnB%d", i), b.Attn.CAttn.B); err != nil {
				return err
			}
		} else {
			L.cAttnB = e.dummyBias
		}
		if L.cProjW, err = mkInit(fmt.Sprintf("cprojW%d", i), b.Attn.CProj.W); err != nil {
			return err
		}
		L.hasCProjB = len(b.Attn.CProj.B) >= h
		if L.hasCProjB {
			if L.cProjB, err = mkInit(fmt.Sprintf("cprojB%d", i), b.Attn.CProj.B); err != nil {
				return err
			}
		} else {
			L.cProjB = e.dummyBias
		}
		if L.fcInW, err = mkInit(fmt.Sprintf("fcInW%d", i), b.FcIn.W); err != nil {
			return err
		}
		L.hasFcInB = len(b.FcIn.B) >= ff
		if L.hasFcInB {
			if L.fcInB, err = mkInit(fmt.Sprintf("fcInB%d", i), b.FcIn.B); err != nil {
				return err
			}
		} else {
			L.fcInB = e.dummyBias
		}
		if L.fcOutW, err = mkInit(fmt.Sprintf("fcOutW%d", i), b.FcOut.W); err != nil {
			return err
		}
		L.hasFcOutB = len(b.FcOut.B) >= h
		if L.hasFcOutB {
			if L.fcOutB, err = mkInit(fmt.Sprintf("fcOutB%d", i), b.FcOut.B); err != nil {
				return err
			}
		} else {
			L.fcOutB = e.dummyBias
		}
		if L.kCache, err = mkStor(fmt.Sprintf("k%d", i), kvElems); err != nil {
			return err
		}
		if L.vCache, err = mkStor(fmt.Sprintf("v%d", i), kvElems); err != nil {
			return err
		}
	}
	return nil
}

// Close releases GPU buffers.
func (e *gpt2Fuse) Close() {
	if e == nil {
		return
	}
	_ = webgpu.WithDevice(func(dev *wgpu.Device, q *wgpu.Queue) error {
		destroy := func(b *wgpu.Buffer) {
			if b != nil {
				b.Destroy()
			}
		}
		destroy(e.stepBuf)
		destroy(e.xBuf)
		destroy(e.normBuf)
		destroy(e.qkvBuf)
		destroy(e.qBuf)
		destroy(e.kBuf)
		destroy(e.vBuf)
		destroy(e.attnBuf)
		destroy(e.projBuf)
		destroy(e.midBuf)
		destroy(e.dummyBias)
		destroy(e.lnfW)
		destroy(e.lnfB)
		for i := range e.layers {
			L := &e.layers[i]
			destroy(L.ln1W)
			destroy(L.ln1B)
			destroy(L.ln2W)
			destroy(L.ln2B)
			destroy(L.cAttnW)
			if L.hasCAttnB {
				destroy(L.cAttnB)
			}
			destroy(L.cProjW)
			if L.hasCProjB {
				destroy(L.cProjB)
			}
			destroy(L.fcInW)
			if L.hasFcInB {
				destroy(L.fcInB)
			}
			destroy(L.fcOutW)
			if L.hasFcOutB {
				destroy(L.fcOutB)
			}
			destroy(L.kCache)
			destroy(L.vCache)
		}
		return nil
	})
	e.layers = nil
}

// ResetPos sets decode position to 0.
func (e *gpt2Fuse) ResetPos() error {
	return webgpu.WithDevice(func(dev *wgpu.Device, q *wgpu.Queue) error {
		q.WriteBuffer(e.stepBuf, 0, packU32(0))
		return nil
	})
}

// DecodeStepFused runs all layers + final LayerNorm; one submit + one readback.
func (e *gpt2Fuse) DecodeStepFused(x []float32) error {
	if e == nil {
		return fmt.Errorf("nil fuse")
	}
	if len(x) < e.hidden {
		return fmt.Errorf("fuse: short x")
	}
	return webgpu.WithDevice(func(dev *wgpu.Device, q *wgpu.Queue) error {
		ctx := &fuseStepCtx{}
		defer ctx.release()
		q.WriteBuffer(e.xBuf, 0, wgpu.ToBytes(x[:e.hidden]))
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
		if err := e.dispatchLN(dev, pass, ctx, e.xBuf, e.lnfW, e.lnfB, e.normBuf); err != nil {
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
		out, err := webgpu.ReadbackF32(dev, q, e.normBuf, e.hidden)
		if err != nil {
			return err
		}
		copy(x, out)
		return nil
	})
}

// DecodeStepNoFinalLN updates residual + KV without ln_f (local mid-tokens).
func (e *gpt2Fuse) DecodeStepNoFinalLN(x []float32) error {
	if e == nil {
		return fmt.Errorf("nil fuse")
	}
	if len(x) < e.hidden {
		return fmt.Errorf("fuse: short x")
	}
	return webgpu.WithDevice(func(dev *wgpu.Device, q *wgpu.Queue) error {
		ctx := &fuseStepCtx{}
		defer ctx.release()
		q.WriteBuffer(e.xBuf, 0, wgpu.ToBytes(x[:e.hidden]))
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
		out, err := webgpu.ReadbackF32(dev, q, e.xBuf, e.hidden)
		if err != nil {
			return err
		}
		copy(x, out)
		return nil
	})
}

// FinalNormGPU applies ln_f.
func (e *gpt2Fuse) FinalNormGPU(x []float32) error {
	return webgpu.WithDevice(func(dev *wgpu.Device, q *wgpu.Queue) error {
		ctx := &fuseStepCtx{}
		defer ctx.release()
		q.WriteBuffer(e.xBuf, 0, wgpu.ToBytes(x[:e.hidden]))
		enc, err := dev.CreateCommandEncoder(nil)
		if err != nil {
			return err
		}
		pass := enc.BeginComputePass(nil)
		if err := e.dispatchLN(dev, pass, ctx, e.xBuf, e.lnfW, e.lnfB, e.normBuf); err != nil {
			pass.End()
			return err
		}
		pass.End()
		cmd, err := enc.Finish(nil)
		if err != nil {
			return err
		}
		q.Submit(cmd)
		out, err := webgpu.ReadbackF32(dev, q, e.normBuf, e.hidden)
		if err != nil {
			return err
		}
		copy(x, out)
		return nil
	})
}

func (e *gpt2Fuse) recordLayer(dev *wgpu.Device, pass *wgpu.ComputePassEncoder, ctx *fuseStepCtx, L *fuseLayerGPU) error {
	h, ff := e.hidden, e.ff
	if err := e.dispatchLN(dev, pass, ctx, e.xBuf, L.ln1W, L.ln1B, e.normBuf); err != nil {
		return err
	}
	if err := e.dispatchGEMV(dev, pass, ctx, e.normBuf, L.cAttnW, L.cAttnB, e.qkvBuf, h, 3*h, L.hasCAttnB); err != nil {
		return err
	}
	if err := e.dispatchCopySlices(dev, pass, ctx); err != nil {
		return err
	}
	if err := e.dispatchRoPE(dev, pass, ctx, e.qBuf); err != nil {
		return err
	}
	if err := e.dispatchRoPE(dev, pass, ctx, e.kBuf); err != nil {
		return err
	}
	if err := e.dispatchWriteKV(dev, pass, ctx, e.kBuf, L.kCache); err != nil {
		return err
	}
	if err := e.dispatchWriteKV(dev, pass, ctx, e.vBuf, L.vCache); err != nil {
		return err
	}
	if err := e.dispatchAttn(dev, pass, ctx, L.kCache, L.vCache); err != nil {
		return err
	}
	if err := e.dispatchGEMV(dev, pass, ctx, e.attnBuf, L.cProjW, L.cProjB, e.projBuf, h, h, L.hasCProjB); err != nil {
		return err
	}
	if err := e.dispatchRes(dev, pass, ctx, e.projBuf, e.xBuf, h); err != nil {
		return err
	}
	if err := e.dispatchLN(dev, pass, ctx, e.xBuf, L.ln2W, L.ln2B, e.normBuf); err != nil {
		return err
	}
	if err := e.dispatchGEMV(dev, pass, ctx, e.normBuf, L.fcInW, L.fcInB, e.midBuf, h, ff, L.hasFcInB); err != nil {
		return err
	}
	if err := e.dispatchGELU(dev, pass, ctx, e.midBuf, ff); err != nil {
		return err
	}
	if err := e.dispatchGEMV(dev, pass, ctx, e.midBuf, L.fcOutW, L.fcOutB, e.projBuf, ff, h, L.hasFcOutB); err != nil {
		return err
	}
	return e.dispatchRes(dev, pass, ctx, e.projBuf, e.xBuf, h)
}

func packU32(v uint32) []byte {
	b := make([]byte, 4)
	binary.LittleEndian.PutUint32(b, v)
	return b
}

func packUniform(u ...uint32) []byte {
	b := make([]byte, len(u)*4)
	for i, v := range u {
		binary.LittleEndian.PutUint32(b[i*4:], v)
	}
	return b
}

func (e *gpt2Fuse) mkUniform(dev *wgpu.Device, ctx *fuseStepCtx, words ...uint32) (*wgpu.Buffer, error) {
	u, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "moss-u", Contents: packUniform(words...),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, err
	}
	ctx.uniforms = append(ctx.uniforms, u)
	return u, nil
}

func (e *gpt2Fuse) bind(dev *wgpu.Device, ctx *fuseStepCtx, pipe *wgpu.ComputePipeline, entries []wgpu.BindGroupEntry) (*wgpu.BindGroup, error) {
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

func (e *gpt2Fuse) dispatchGEMV(dev *wgpu.Device, pass *wgpu.ComputePassEncoder, ctx *fuseStepCtx, x, w, b, y *wgpu.Buffer, in, out int, hasBias bool) error {
	hb := uint32(0)
	if hasBias {
		hb = 1
	}
	u, err := e.mkUniform(dev, ctx, uint32(in), uint32(out), hb, 0)
	if err != nil {
		return err
	}
	bg, err := e.bind(dev, ctx, e.pipeGEMV, []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: u, Size: u.GetSize()},
		{Binding: 1, Buffer: x, Size: x.GetSize()},
		{Binding: 2, Buffer: w, Size: w.GetSize()},
		{Binding: 3, Buffer: b, Size: b.GetSize()},
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

func (e *gpt2Fuse) dispatchLN(dev *wgpu.Device, pass *wgpu.ComputePassEncoder, ctx *fuseStepCtx, in, w, b, out *wgpu.Buffer) error {
	u, err := e.mkUniform(dev, ctx, uint32(e.hidden), e.epsBits, 0, 0)
	if err != nil {
		return err
	}
	bg, err := e.bind(dev, ctx, e.pipeLN, []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: u, Size: u.GetSize()},
		{Binding: 1, Buffer: in, Size: in.GetSize()},
		{Binding: 2, Buffer: w, Size: w.GetSize()},
		{Binding: 3, Buffer: b, Size: b.GetSize()},
		{Binding: 4, Buffer: out, Size: out.GetSize()},
	})
	if err != nil {
		return err
	}
	pass.SetPipeline(e.pipeLN)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(1, 1, 1)
	return nil
}

func (e *gpt2Fuse) dispatchRes(dev *wgpu.Device, pass *wgpu.ComputePassEncoder, ctx *fuseStepCtx, addend, inout *wgpu.Buffer, n int) error {
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

func (e *gpt2Fuse) dispatchGELU(dev *wgpu.Device, pass *wgpu.ComputePassEncoder, ctx *fuseStepCtx, buf *wgpu.Buffer, n int) error {
	u, err := e.mkUniform(dev, ctx, uint32(n), 0, 0, 0)
	if err != nil {
		return err
	}
	bg, err := e.bind(dev, ctx, e.pipeGELU, []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: u, Size: u.GetSize()},
		{Binding: 1, Buffer: buf, Size: buf.GetSize()},
	})
	if err != nil {
		return err
	}
	pass.SetPipeline(e.pipeGELU)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(tiling.GPUWorkgroupsX(n, 64), 1, 1)
	return nil
}

func (e *gpt2Fuse) dispatchRoPE(dev *wgpu.Device, pass *wgpu.ComputePassEncoder, ctx *fuseStepCtx, x *wgpu.Buffer) error {
	u, err := e.mkUniform(dev, ctx, uint32(e.heads), uint32(e.headDim), math.Float32bits(e.ropeBase), 0)
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
	pass.DispatchWorkgroups(tiling.GPUWorkgroupsX(e.heads, 64), 1, 1)
	return nil
}

func (e *gpt2Fuse) dispatchWriteKV(dev *wgpu.Device, pass *wgpu.ComputePassEncoder, ctx *fuseStepCtx, src, cache *wgpu.Buffer) error {
	u, err := e.mkUniform(dev, ctx, uint32(e.heads), uint32(e.headDim), uint32(e.maxSeq), 0)
	if err != nil {
		return err
	}
	bg, err := e.bind(dev, ctx, e.pipeWKV, []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: u, Size: u.GetSize()},
		{Binding: 1, Buffer: e.stepBuf, Size: e.stepBuf.GetSize()},
		{Binding: 2, Buffer: src, Size: src.GetSize()},
		{Binding: 3, Buffer: cache, Size: cache.GetSize()},
	})
	if err != nil {
		return err
	}
	pass.SetPipeline(e.pipeWKV)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(tiling.GPUWorkgroupsX(e.heads, 64), 1, 1)
	return nil
}

func (e *gpt2Fuse) dispatchAttn(dev *wgpu.Device, pass *wgpu.ComputePassEncoder, ctx *fuseStepCtx, kCache, vCache *wgpu.Buffer) error {
	var scaleBits uint32
	if e.scaleAttn {
		scaleBits = math.Float32bits(float32(1 / math.Sqrt(float64(e.headDim))))
	} else {
		scaleBits = math.Float32bits(1)
	}
	u, err := e.mkUniform(dev, ctx, uint32(e.heads), uint32(e.headDim), uint32(e.maxSeq), scaleBits)
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
	pass.DispatchWorkgroups(uint32(e.heads), 1, 1)
	return nil
}

func (e *gpt2Fuse) dispatchInc(dev *wgpu.Device, pass *wgpu.ComputePassEncoder, ctx *fuseStepCtx) error {
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

func (e *gpt2Fuse) dispatchCopySlices(dev *wgpu.Device, pass *wgpu.ComputePassEncoder, ctx *fuseStepCtx) error {
	h := e.hidden
	copyOne := func(off int, dst *wgpu.Buffer) error {
		u, err := e.mkUniform(dev, ctx, uint32(h), uint32(off), 0, 0)
		if err != nil {
			return err
		}
		bg, err := e.bind(dev, ctx, e.pipeCopy, []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: u, Size: u.GetSize()},
			{Binding: 1, Buffer: e.qkvBuf, Size: e.qkvBuf.GetSize()},
			{Binding: 2, Buffer: dst, Size: dst.GetSize()},
		})
		if err != nil {
			return err
		}
		pass.SetPipeline(e.pipeCopy)
		pass.SetBindGroup(0, bg, nil)
		pass.DispatchWorkgroups(tiling.GPUWorkgroupsX(h, 64), 1, 1)
		return nil
	}
	if err := copyOne(0, e.qBuf); err != nil {
		return err
	}
	if err := copyOne(h, e.kBuf); err != nil {
		return err
	}
	return copyOne(2*h, e.vBuf)
}
