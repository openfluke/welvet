package fusedgpu

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/openfluke/webgpu/wgpu"
)

type q4GPU struct {
	scales, weights *wgpu.Buffer
	rows, cols      int
}

type blockGPU struct {
	attnNorm, mlpNorm *wgpu.Buffer
	q, k, v, o        q4GPU
	gate, up, down    q4GPU
	kCache, vCache    *wgpu.Buffer
}

type engine struct {
	instance *wgpu.Instance
	adapter  *wgpu.Adapter
	device   *wgpu.Device
	queue    *wgpu.Queue

	pipe map[string]*wgpu.ComputePipeline
	bg   map[string]*wgpu.BindGroup // stable cached bind groups
	owned []*wgpu.Buffer           // all device buffers for release()

	embed, finalNorm *wgpu.Buffer
	lmScales, lmW    *wgpu.Buffer
	blocks           []blockGPU

	// GPU control / scratch
	step        *wgpu.Buffer // [pos, outCount]
	token       *wgpu.Buffer // current input token
	promptBuf   *wgpu.Buffer
	histBuf     *wgpu.Buffer // generated tokens
	stagingHist *wgpu.Buffer

	hidden, normed         *wgpu.Buffer
	qkvBuf, attnOut        *wgpu.Buffer // qkv = [Q|K|V]
	qOff, kOff, vOff       uint64
	qBytes, kBytes, vBytes uint64
	inter, logits, outTok  *wgpu.Buffer

	// Stable uniforms (shape constants — no per-step pos)
	uGemvQDimH, uGemvHInter *wgpu.Buffer
	uGemvVocabH             *wgpu.Buffer
	uQKV                    *wgpu.Buffer
	uSwiglu, uRMS, uResidH  *wgpu.Buffer
	uRopeQ, uRopeK, uAttn, uKV *wgpu.Buffer
	uEmbed, uArgMax         *wgpu.Buffer

	m             *modelCPU
	pos           int
	stagingLogits *wgpu.Buffer
}

func newEngine(m *modelCPU) (*engine, error) {
	e := &engine{m: m, pipe: map[string]*wgpu.ComputePipeline{}, bg: map[string]*wgpu.BindGroup{}}
	inst, adapt, device, queue, _, err := acquireDevice()
	if err != nil {
		return nil, err
	}
	e.instance = inst
	e.adapter = adapt
	e.device = device
	e.queue = queue

	shaders := map[string]string{
		"q4gemv":  shaderQ4GEMV,
		"qkv":     shaderQ4GEMV_QKV,
		"swiglu":  shaderQ4SwiGLUFused,
		"rmsnorm": shaderRMSNorm,
		"resid":   shaderResidual,
		"rope":    shaderRoPE,
		"attn":    shaderAttnDecode,
		"kv":      shaderKVUpdate,
		"embed":   shaderEmbed,
		"argmax":  shaderArgMax,
		"advance": shaderAdvance,
		"embed_p": shaderEmbedPrompt,
		"inc_pos": shaderIncPos,
	}
	for name, src := range shaders {
		p, err := e.createPipeline(src)
		if err != nil {
			e.releaseModelGPU()
			return nil, fmt.Errorf("pipeline %s: %w", name, err)
		}
		e.pipe[name] = p
	}

	if err := e.uploadModel(); err != nil {
		e.releaseModelGPU()
		return nil, err
	}
	if err := e.allocScratch(); err != nil {
		e.releaseModelGPU()
		return nil, err
	}
	e.dropHostWeightPayloads()
	e.initUniforms()
	e.buildBindGroups()
	fmt.Println("✅ GPU engine ready (chunked on-device decode)")
	return e, nil
}

// dropHostWeightPayloads frees CPU weight copies after GPU upload.
// Shape fields on e.m stay; only large payloads are cleared.
func (e *engine) dropHostWeightPayloads() {
	if e == nil || e.m == nil {
		return
	}
	m := e.m
	m.embed, m.finalNorm = nil, nil
	m.lmScales, m.lmPacked = nil, nil
	for i := range m.blocks {
		b := &m.blocks[i]
		b.attnNorm.w, b.mlpNorm.w = nil, nil
		clearQ4Host(&b.q)
		clearQ4Host(&b.k)
		clearQ4Host(&b.v)
		clearQ4Host(&b.o)
		clearQ4Host(&b.gate)
		clearQ4Host(&b.up)
		clearQ4Host(&b.down)
	}
	runtime.GC()
}

func clearQ4Host(q *q4Mat) {
	if q == nil {
		return
	}
	q.scales, q.packed = nil, nil
}

func (e *engine) createPipeline(wgsl string) (*wgpu.ComputePipeline, error) {
	mod, err := e.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: wgsl},
	})
	if err != nil {
		return nil, err
	}
	defer mod.Release()
	return e.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Compute: wgpu.ProgrammableStageDescriptor{Module: mod, EntryPoint: "main"},
	})
}

func (e *engine) mkBuf(label string, size uint64, usage wgpu.BufferUsage, data []byte) (*wgpu.Buffer, error) {
	if size < 64 {
		size = 64
	}
	if size%16 != 0 {
		size = (size + 15) &^ 15
	}
	if usage&wgpu.BufferUsageMapRead != 0 {
		usage |= wgpu.BufferUsageCopyDst
	} else if usage&wgpu.BufferUsageMapWrite != 0 {
		usage |= wgpu.BufferUsageCopySrc
	} else {
		usage |= wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc
	}
	var b *wgpu.Buffer
	var err error
	if len(data) > 0 {
		b, err = e.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
			Label: label, Contents: data, Usage: usage,
		})
	} else {
		b, err = e.device.CreateBuffer(&wgpu.BufferDescriptor{Label: label, Size: size, Usage: usage})
	}
	if err != nil || b == nil {
		return nil, fmt.Errorf("CreateBuffer %s: %w", label, err)
	}
	e.owned = append(e.owned, b)
	return b, nil
}

func (e *engine) uploadQ4(label string, m q4Mat) (q4GPU, error) {
	s, err := e.mkBuf(label+"_s", uint64(len(m.scales)*4), wgpu.BufferUsageStorage, f32Bytes(m.scales))
	if err != nil {
		return q4GPU{}, err
	}
	w, err := e.mkBuf(label+"_w", uint64(len(m.packed)*4), wgpu.BufferUsageStorage, u32Bytes(m.packed))
	if err != nil {
		return q4GPU{}, err
	}
	return q4GPU{scales: s, weights: w, rows: m.rows, cols: m.cols}, nil
}

func (e *engine) uploadModel() error {
	m := e.m
	var err error
	if e.embed, err = e.mkBuf("embed", uint64(len(m.embed)*4), wgpu.BufferUsageStorage, f32Bytes(m.embed)); err != nil {
		return err
	}
	m.embed = nil
	if e.finalNorm, err = e.mkBuf("fnorm", uint64(len(m.finalNorm)*4), wgpu.BufferUsageStorage, f32Bytes(m.finalNorm)); err != nil {
		return err
	}
	m.finalNorm = nil
	if e.lmScales, err = e.mkBuf("lm_s", uint64(len(m.lmScales)*4), wgpu.BufferUsageStorage, f32Bytes(m.lmScales)); err != nil {
		return err
	}
	m.lmScales = nil
	if e.lmW, err = e.mkBuf("lm_w", uint64(len(m.lmPacked)*4), wgpu.BufferUsageStorage, u32Bytes(m.lmPacked)); err != nil {
		return err
	}
	m.lmPacked = nil

	e.blocks = make([]blockGPU, m.layers)
	kvBytes := uint64(m.kvHeads * m.maxSeq * m.headDim * 4)
	for i := range m.blocks {
		b := &m.blocks[i]
		g := &e.blocks[i]
		if g.attnNorm, err = e.mkBuf(fmt.Sprintf("n1_%d", i), uint64(len(b.attnNorm.w)*4), wgpu.BufferUsageStorage, f32Bytes(b.attnNorm.w)); err != nil {
			return err
		}
		if g.mlpNorm, err = e.mkBuf(fmt.Sprintf("n2_%d", i), uint64(len(b.mlpNorm.w)*4), wgpu.BufferUsageStorage, f32Bytes(b.mlpNorm.w)); err != nil {
			return err
		}
		if g.q, err = e.uploadQ4(fmt.Sprintf("q_%d", i), b.q); err != nil {
			return err
		}
		clearQ4Host(&b.q)
		if g.k, err = e.uploadQ4(fmt.Sprintf("k_%d", i), b.k); err != nil {
			return err
		}
		clearQ4Host(&b.k)
		if g.v, err = e.uploadQ4(fmt.Sprintf("v_%d", i), b.v); err != nil {
			return err
		}
		clearQ4Host(&b.v)
		if g.o, err = e.uploadQ4(fmt.Sprintf("o_%d", i), b.o); err != nil {
			return err
		}
		clearQ4Host(&b.o)
		if g.gate, err = e.uploadQ4(fmt.Sprintf("g_%d", i), b.gate); err != nil {
			return err
		}
		clearQ4Host(&b.gate)
		if g.up, err = e.uploadQ4(fmt.Sprintf("u_%d", i), b.up); err != nil {
			return err
		}
		clearQ4Host(&b.up)
		if g.down, err = e.uploadQ4(fmt.Sprintf("d_%d", i), b.down); err != nil {
			return err
		}
		clearQ4Host(&b.down)
		if g.kCache, err = e.mkBuf(fmt.Sprintf("kc_%d", i), kvBytes, wgpu.BufferUsageStorage, nil); err != nil {
			return err
		}
		if g.vCache, err = e.mkBuf(fmt.Sprintf("vc_%d", i), kvBytes, wgpu.BufferUsageStorage, nil); err != nil {
			return err
		}
	}
	return nil
}

func (e *engine) allocScratch() error {
	m := e.m
	H := uint64(m.hidden * 4)
	var err error
	if e.step, err = e.mkBuf("step", 64, wgpu.BufferUsageStorage, nil); err != nil {
		return err
	}
	if e.token, err = e.mkBuf("token", 64, wgpu.BufferUsageStorage, nil); err != nil {
		return err
	}
	if e.promptBuf, err = e.mkBuf("prompt", uint64(m.maxSeq*4), wgpu.BufferUsageStorage, nil); err != nil {
		return err
	}
	if e.histBuf, err = e.mkBuf("hist", uint64(m.maxSeq*4), wgpu.BufferUsageStorage, nil); err != nil {
		return err
	}
	if e.stagingHist, err = e.mkBuf("stageHist", uint64(m.maxSeq*4), wgpu.BufferUsageMapRead, nil); err != nil {
		return err
	}
	if e.hidden, err = e.mkBuf("h", H, wgpu.BufferUsageStorage, nil); err != nil {
		return err
	}
	if e.normed, err = e.mkBuf("norm", H, wgpu.BufferUsageStorage, nil); err != nil {
		return err
	}
	e.qBytes = uint64(m.qDim * 4)
	e.kBytes = uint64(m.kvDim * 4)
	e.vBytes = uint64(m.kvDim * 4)
	e.qOff, e.kOff, e.vOff = 0, e.qBytes, e.qBytes+e.kBytes
	if e.qkvBuf, err = e.mkBuf("qkv", e.qBytes+e.kBytes+e.vBytes, wgpu.BufferUsageStorage, nil); err != nil {
		return err
	}
	if e.attnOut, err = e.mkBuf("ao", e.qBytes, wgpu.BufferUsageStorage, nil); err != nil {
		return err
	}
	if e.inter, err = e.mkBuf("inter", uint64(m.intermediate*4), wgpu.BufferUsageStorage, nil); err != nil {
		return err
	}
	if e.logits, err = e.mkBuf("logits", uint64(m.vocab*4), wgpu.BufferUsageStorage, nil); err != nil {
		return err
	}
	if e.outTok, err = e.mkBuf("outTok", 64, wgpu.BufferUsageStorage, nil); err != nil {
		return err
	}
	if e.stagingLogits, err = e.mkBuf("stageLogits", uint64(m.vocab*4), wgpu.BufferUsageMapRead, nil); err != nil {
		return err
	}
	return nil
}

func (e *engine) uniInit(label string, bytes []byte) *wgpu.Buffer {
	b, err := e.mkBuf(label, 256, wgpu.BufferUsageUniform, bytes)
	if err != nil {
		panic(err) // uniforms are tiny; real OOM already hit during upload
	}
	return b
}

func (e *engine) initUniforms() {
	m := e.m
	e.uQKV = e.uniInit("uQKV", packU32(uint32(m.hidden), uint32(m.qDim), uint32(m.kvDim), 0))
	e.uGemvQDimH = e.uniInit("uQH", packU32(uint32(m.qDim), uint32(m.hidden), 0, 0))
	e.uGemvHInter = e.uniInit("uHI", packU32(uint32(m.intermediate), uint32(m.hidden), 0, 0))
	e.uGemvVocabH = e.uniInit("uVH", packU32(uint32(m.hidden), uint32(m.vocab), 0, 0))
	e.uSwiglu = e.uniInit("uSW", packU32(uint32(m.hidden), uint32(m.intermediate), 0, 0))
	e.uRMS = e.uniInit("uRMS", packMix(uint32(m.hidden), m.eps, 0, 0))
	e.uResidH = e.uniInit("uRH", packU32(uint32(m.hidden), 0, 0, 0))
	e.uRopeQ = e.uniInit("uRQ", packU32(uint32(m.heads), uint32(m.headDim), mathFloat32bits(m.ropeTheta), 0))
	e.uRopeK = e.uniInit("uRK", packU32(uint32(m.kvHeads), uint32(m.headDim), mathFloat32bits(m.ropeTheta), 0))
	e.uAttn = e.uniInit("uAT", packU32(uint32(m.heads), uint32(m.kvHeads), uint32(m.headDim), uint32(m.maxSeq)))
	e.uKV = e.uniInit("uKV", packU32(uint32(m.kvDim), uint32(m.maxSeq), uint32(m.headDim), 0))
	e.uEmbed = e.uniInit("uEM", packU32(uint32(m.hidden), uint32(m.vocab), 0, 0))
	e.uArgMax = e.uniInit("uAM", packU32(uint32(m.vocab), 0, 0, 0))
}

type bufSlice struct {
	buf          *wgpu.Buffer
	offset, size uint64
}

func (e *engine) mkBG(key string, pipe *wgpu.ComputePipeline, slices ...bufSlice) *wgpu.BindGroup {
	if bg, ok := e.bg[key]; ok {
		return bg
	}
	entries := make([]wgpu.BindGroupEntry, len(slices))
	for i, s := range slices {
		sz := s.size
		if sz == 0 {
			sz = wgpu.WholeSize
		}
		entries[i] = wgpu.BindGroupEntry{Binding: uint32(i), Buffer: s.buf, Offset: s.offset, Size: sz}
	}
	bg, err := e.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: pipe.GetBindGroupLayout(0), Entries: entries,
	})
	if err != nil {
		panic(err)
	}
	e.bg[key] = bg
	return bg
}

func whole(b *wgpu.Buffer) bufSlice { return bufSlice{buf: b} }

func (e *engine) buildBindGroups() {
	m := e.m
	p := e.pipe
	qView := bufSlice{e.qkvBuf, e.qOff, e.qBytes}
	kView := bufSlice{e.qkvBuf, e.kOff, e.kBytes}
	vView := bufSlice{e.qkvBuf, e.vOff, e.vBytes}

	e.mkBG("embed_tok", p["embed"], whole(e.uEmbed), whole(e.token), whole(e.embed), whole(e.hidden))
	e.mkBG("embed_p", p["embed_p"], whole(e.uEmbed), whole(e.step), whole(e.promptBuf), whole(e.embed), whole(e.hidden))
	e.mkBG("argmax", p["argmax"], whole(e.uArgMax), whole(e.logits), whole(e.outTok))
	e.mkBG("advance", p["advance"], whole(e.step), whole(e.outTok), whole(e.histBuf), whole(e.token))
	e.mkBG("inc_pos", p["inc_pos"], whole(e.step))
	e.mkBG("fnorm", p["rmsnorm"], whole(e.uRMS), whole(e.hidden), whole(e.finalNorm), whole(e.normed))
	e.mkBG("lm", p["q4gemv"], whole(e.uGemvVocabH), whole(e.normed), whole(e.lmScales), whole(e.lmW), whole(e.logits))

	for i := range e.blocks {
		b := &e.blocks[i]
		tag := fmt.Sprintf("L%d", i)
		e.mkBG(tag+"_rms1", p["rmsnorm"], whole(e.uRMS), whole(e.hidden), whole(b.attnNorm), whole(e.normed))
		e.mkBG(tag+"_qkv", p["qkv"], whole(e.uQKV), whole(e.normed),
			whole(b.q.scales), whole(b.q.weights),
			whole(b.k.scales), whole(b.k.weights),
			whole(b.v.scales), whole(b.v.weights),
			whole(e.qkvBuf))
		e.mkBG(tag+"_ropeq", p["rope"], whole(e.uRopeQ), whole(e.step), qView)
		e.mkBG(tag+"_ropek", p["rope"], whole(e.uRopeK), whole(e.step), kView)
		e.mkBG(tag+"_kv", p["kv"], whole(e.uKV), whole(e.step), kView, vView, whole(b.kCache), whole(b.vCache))
		e.mkBG(tag+"_attn", p["attn"], whole(e.uAttn), whole(e.step), qView, whole(b.kCache), whole(b.vCache), whole(e.attnOut))
		e.mkBG(tag+"_o", p["q4gemv"], whole(e.uGemvQDimH), whole(e.attnOut), whole(b.o.scales), whole(b.o.weights), whole(e.normed))
		e.mkBG(tag+"_r1", p["resid"], whole(e.uResidH), whole(e.normed), whole(e.hidden))
		e.mkBG(tag+"_rms2", p["rmsnorm"], whole(e.uRMS), whole(e.hidden), whole(b.mlpNorm), whole(e.normed))
		e.mkBG(tag+"_sw", p["swiglu"], whole(e.uSwiglu), whole(e.normed),
			whole(b.gate.scales), whole(b.gate.weights), whole(b.up.scales), whole(b.up.weights), whole(e.inter))
		e.mkBG(tag+"_d", p["q4gemv"], whole(e.uGemvHInter), whole(e.inter), whole(b.down.scales), whole(b.down.weights), whole(e.normed))
		e.mkBG(tag+"_r2", p["resid"], whole(e.uResidH), whole(e.normed), whole(e.hidden))
	}
	_ = m
}

func minU64(a, b uint64) uint64 {
	if a < b {
		return a
	}
	return b
}

func f32Bytes(v []float32) []byte {
	if len(v) == 0 {
		return nil
	}
	return wgpu.ToBytes(v)
}

func u32Bytes(v []uint32) []byte {
	if len(v) == 0 {
		return nil
	}
	return wgpu.ToBytes(v)
}

func packU32(vals ...uint32) []byte {
	out := make([]byte, len(vals)*4)
	for i, v := range vals {
		*(*uint32)(unsafe.Pointer(&out[i*4])) = v
	}
	return out
}

func packMix(u0 uint32, f1 float32, u2, u3 uint32) []byte {
	b := make([]byte, 16)
	*(*uint32)(unsafe.Pointer(&b[0])) = u0
	*(*float32)(unsafe.Pointer(&b[4])) = f1
	*(*uint32)(unsafe.Pointer(&b[8])) = u2
	*(*uint32)(unsafe.Pointer(&b[12])) = u3
	return b
}

func mathFloat32bits(f float32) uint32 {
	b := make([]byte, 4)
	*(*float32)(unsafe.Pointer(&b[0])) = f
	return *(*uint32)(unsafe.Pointer(&b[0]))
}
