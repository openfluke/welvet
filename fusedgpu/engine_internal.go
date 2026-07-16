package fusedgpu

import (
	"fmt"
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

	embed, finalNorm *wgpu.Buffer
	lmScales, lmW    *wgpu.Buffer
	blocks           []blockGPU

	// GPU control / scratch
	step       *wgpu.Buffer // [pos, outCount]
	token      *wgpu.Buffer // current input token
	promptBuf  *wgpu.Buffer
	histBuf    *wgpu.Buffer // generated tokens
	stagingHist *wgpu.Buffer

	hidden, normed        *wgpu.Buffer
	qkvBuf, attnOut       *wgpu.Buffer // qkv = [Q|K|V]
	qOff, kOff, vOff      uint64
	qBytes, kBytes, vBytes uint64
	inter, logits, outTok *wgpu.Buffer

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
	inst := wgpu.CreateInstance(&wgpu.InstanceDescriptor{Backends: wgpu.InstanceBackendVulkan})
	if inst == nil {
		return nil, fmt.Errorf("CreateInstance failed")
	}
	e.instance = inst

	adapter, err := inst.RequestAdapter(&wgpu.RequestAdapterOptions{
		PowerPreference: wgpu.PowerPreferenceHighPerformance,
	})
	if err != nil || adapter == nil {
		return nil, fmt.Errorf("RequestAdapter: %w", err)
	}
	e.adapter = adapter
	info := e.adapter.GetInfo()
	fmt.Printf("Adapter: %s [%v]\n", info.Name, info.BackendType)

	limits := e.adapter.GetLimits().Limits
	limits.MaxStorageBufferBindingSize = minU64(1<<30, limits.MaxStorageBufferBindingSize)
	limits.MaxBufferSize = minU64(2<<30, limits.MaxBufferSize)
	if limits.MaxStorageBuffersPerShaderStage < 12 {
		limits.MaxStorageBuffersPerShaderStage = 12
	}

	device, err := e.adapter.RequestDevice(&wgpu.DeviceDescriptor{
		RequiredLimits: &wgpu.RequiredLimits{Limits: limits},
	})
	if err != nil || device == nil {
		return nil, fmt.Errorf("RequestDevice: %w", err)
	}
	e.device = device
	e.queue = e.device.GetQueue()

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
			return nil, fmt.Errorf("pipeline %s: %w", name, err)
		}
		e.pipe[name] = p
	}

	if err := e.uploadModel(); err != nil {
		return nil, err
	}
	e.allocScratch()
	e.initUniforms()
	e.buildBindGroups()
	fmt.Println("✅ GPU engine ready (chunked on-device decode)")
	return e, nil
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

func (e *engine) mkBuf(label string, size uint64, usage wgpu.BufferUsage, data []byte) *wgpu.Buffer {
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
	if len(data) > 0 {
		b, err := e.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
			Label: label, Contents: data, Usage: usage,
		})
		if err != nil || b == nil {
			panic(fmt.Sprintf("CreateBufferInit %s: %v", label, err))
		}
		return b
	}
	b, err := e.device.CreateBuffer(&wgpu.BufferDescriptor{Label: label, Size: size, Usage: usage})
	if err != nil || b == nil {
		panic(fmt.Sprintf("CreateBuffer %s: %v", label, err))
	}
	return b
}

func (e *engine) uploadQ4(label string, m q4Mat) q4GPU {
	return q4GPU{
		scales:  e.mkBuf(label+"_s", uint64(len(m.scales)*4), wgpu.BufferUsageStorage, f32Bytes(m.scales)),
		weights: e.mkBuf(label+"_w", uint64(len(m.packed)*4), wgpu.BufferUsageStorage, u32Bytes(m.packed)),
		rows:    m.rows,
		cols:    m.cols,
	}
}

func (e *engine) uploadModel() error {
	m := e.m
	e.embed = e.mkBuf("embed", uint64(len(m.embed)*4), wgpu.BufferUsageStorage, f32Bytes(m.embed))
	e.finalNorm = e.mkBuf("fnorm", uint64(len(m.finalNorm)*4), wgpu.BufferUsageStorage, f32Bytes(m.finalNorm))
	e.lmScales = e.mkBuf("lm_s", uint64(len(m.lmScales)*4), wgpu.BufferUsageStorage, f32Bytes(m.lmScales))
	e.lmW = e.mkBuf("lm_w", uint64(len(m.lmPacked)*4), wgpu.BufferUsageStorage, u32Bytes(m.lmPacked))

	e.blocks = make([]blockGPU, m.layers)
	kvBytes := uint64(m.kvHeads * m.maxSeq * m.headDim * 4)
	for i := range m.blocks {
		b := &m.blocks[i]
		g := &e.blocks[i]
		g.attnNorm = e.mkBuf(fmt.Sprintf("n1_%d", i), uint64(len(b.attnNorm.w)*4), wgpu.BufferUsageStorage, f32Bytes(b.attnNorm.w))
		g.mlpNorm = e.mkBuf(fmt.Sprintf("n2_%d", i), uint64(len(b.mlpNorm.w)*4), wgpu.BufferUsageStorage, f32Bytes(b.mlpNorm.w))
		g.q = e.uploadQ4(fmt.Sprintf("q_%d", i), b.q)
		g.k = e.uploadQ4(fmt.Sprintf("k_%d", i), b.k)
		g.v = e.uploadQ4(fmt.Sprintf("v_%d", i), b.v)
		g.o = e.uploadQ4(fmt.Sprintf("o_%d", i), b.o)
		g.gate = e.uploadQ4(fmt.Sprintf("g_%d", i), b.gate)
		g.up = e.uploadQ4(fmt.Sprintf("u_%d", i), b.up)
		g.down = e.uploadQ4(fmt.Sprintf("d_%d", i), b.down)
		g.kCache = e.mkBuf(fmt.Sprintf("kc_%d", i), kvBytes, wgpu.BufferUsageStorage, nil)
		g.vCache = e.mkBuf(fmt.Sprintf("vc_%d", i), kvBytes, wgpu.BufferUsageStorage, nil)
	}
	return nil
}

func (e *engine) allocScratch() {
	m := e.m
	H := uint64(m.hidden * 4)
	e.step = e.mkBuf("step", 64, wgpu.BufferUsageStorage, nil)
	e.token = e.mkBuf("token", 64, wgpu.BufferUsageStorage, nil)
	e.promptBuf = e.mkBuf("prompt", uint64(m.maxSeq*4), wgpu.BufferUsageStorage, nil)
	e.histBuf = e.mkBuf("hist", uint64(m.maxSeq*4), wgpu.BufferUsageStorage, nil)
	e.stagingHist = e.mkBuf("stageHist", uint64(m.maxSeq*4), wgpu.BufferUsageMapRead, nil)
	e.hidden = e.mkBuf("h", H, wgpu.BufferUsageStorage, nil)
	e.normed = e.mkBuf("norm", H, wgpu.BufferUsageStorage, nil)
	e.qBytes = uint64(m.qDim * 4)
	e.kBytes = uint64(m.kvDim * 4)
	e.vBytes = uint64(m.kvDim * 4)
	e.qOff, e.kOff, e.vOff = 0, e.qBytes, e.qBytes+e.kBytes
	e.qkvBuf = e.mkBuf("qkv", e.qBytes+e.kBytes+e.vBytes, wgpu.BufferUsageStorage, nil)
	e.attnOut = e.mkBuf("ao", e.qBytes, wgpu.BufferUsageStorage, nil)
	e.inter = e.mkBuf("inter", uint64(m.intermediate*4), wgpu.BufferUsageStorage, nil)
	e.logits = e.mkBuf("logits", uint64(m.vocab*4), wgpu.BufferUsageStorage, nil)
	e.outTok = e.mkBuf("outTok", 64, wgpu.BufferUsageStorage, nil)
	e.stagingLogits = e.mkBuf("stageLogits", uint64(m.vocab*4), wgpu.BufferUsageMapRead, nil)
}

func (e *engine) uniInit(label string, bytes []byte) *wgpu.Buffer {
	return e.mkBuf(label, 256, wgpu.BufferUsageUniform, bytes)
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
