package fusedgpu

import (
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"time"
	"unsafe"

	"github.com/openfluke/webgpu/wgpu"
)

type binGPU struct {
	scales, weights *wgpu.Buffer
	rows, cols      int
}

type hybridBlockGPU struct {
	layerType string
	attnNorm  *wgpu.Buffer
	ffnNorm   *wgpu.Buffer
	gate, up, down binGPU

	q, k, v, o     binGPU
	qNorm, kNorm   *wgpu.Buffer
	kCache, vCache *wgpu.Buffer
	outputGate     bool
	partialRotary  float32
	ropeTheta      float32
	numHeads       int
	numKVHeads     int
	headDim        int
	qRows          int

	gdnQKV, gdnZ, gdnB, gdnA, gdnOut binGPU
	gdnConv, gdnALog, gdnDtBias      *wgpu.Buffer
	gdnNorm                          *wgpu.Buffer
	gdnState, gdnConvState           *wgpu.Buffer
	numKeyHeads, numValueHeads       int
	keyHeadDim, valueHeadDim         int
	convKernel                       int
}

type hybridEngine struct {
	instance *wgpu.Instance
	adapter  *wgpu.Adapter
	device   *wgpu.Device
	queue    *wgpu.Queue

	pipe  map[string]*wgpu.ComputePipeline
	bg    map[string]*wgpu.BindGroup
	owned []*wgpu.Buffer

	spec *HybridSpec

	embed, lmHead binGPU
	finalNorm     *wgpu.Buffer
	blocks        []hybridBlockGPU

	step, token         *wgpu.Buffer
	hidden, normed, mix *wgpu.Buffer
	inter, upBuf        *wgpu.Buffer
	logits              *wgpu.Buffer
	stagingLogits       *wgpu.Buffer

	qGate, qBuf, gateBuf, kBuf, vBuf, attnOut *wgpu.Buffer

	gdnQKV, gdnZ, gdnBetaRaw, gdnARaw *wgpu.Buffer
	gdnMixed, gdnQRep, gdnKRep        *wgpu.Buffer
	gdnBeta, gdnG, gdnCore            *wgpu.Buffer

	uRMS, uResidH, uSwiglu   *wgpu.Buffer
	uEmbed                   *wgpu.Buffer
	uGemvVocabH              *wgpu.Buffer
	uGemvHInter, uGemvInterH *wgpu.Buffer
	uZero                    *wgpu.Buffer

	hiddenN, vocabN, interN, maxSeq int
	eps                             float32
	pos                             int

	maxQDim, maxKVDim, maxQGate int
	maxConvDim, maxValDim       int
	maxNumV, maxHdK, maxHdV     int
	maxNumK                     int
	maxConvHist                 int
}

func newHybridEngine(spec *HybridSpec) (*hybridEngine, error) {
	e := &hybridEngine{
		pipe:    map[string]*wgpu.ComputePipeline{},
		bg:      map[string]*wgpu.BindGroup{},
		spec:    spec,
		hiddenN: spec.Hidden,
		vocabN:  spec.Vocab,
		interN:  spec.Intermediate,
		maxSeq:  spec.MaxSeq,
		eps:     spec.Eps,
	}
	if e.maxSeq <= 0 {
		e.maxSeq = 256
	}
	if e.eps <= 0 {
		e.eps = 1e-6
	}
	e.deriveMaxDims(spec)

	inst, adapt, device, queue, _, err := acquireDevice()
	if err != nil {
		return nil, err
	}
	e.instance = inst
	e.adapter = adapt
	e.device = device
	e.queue = queue

	shaders := map[string]string{
		"bingemv":    shaderBinG128GEMV,
		"binembed":   shaderBinEmbed,
		"rmsnorm":    shaderHybridRMS,
		"resid":      shaderHybridResid,
		"swiglu":     shaderHybridSwiGLU,
		"gdn_conv":   shaderGDNConv,
		"gdn_prepqk": shaderGDNPrepQK,
		"gdn_betag":  shaderGDNBetaG,
		"gdn_step":   shaderGDNStep,
		"gdn_gnorm":  shaderGDNGateNorm,
		"head_rms":   shaderHeadRMS,
		"split_qg":   shaderSplitQGate,
		"prope":      shaderPartialRoPE,
		"kv":         shaderHybridKVUpdate,
		"attn":       shaderHybridAttn,
		"outgate":    shaderOutGate,
		"inc_pos":    shaderIncPosHybrid,
		"zero":       shaderZeroF32,
	}
	for name, src := range shaders {
		p, err := e.createPipeline(src)
		if err != nil {
			e.release()
			return nil, fmt.Errorf("pipeline %s: %w", name, err)
		}
		e.pipe[name] = p
	}

	if err := e.uploadAll(spec); err != nil {
		e.release()
		return nil, err
	}
	if err := e.allocScratch(); err != nil {
		e.release()
		return nil, err
	}
	e.initUniforms()
	e.buildBindGroups()
	nbytes := e.estimateVRAM()
	fmt.Printf("✅ Hybrid GPU fuse ready (BinaryG128 full on-device, ~%.1f GiB weights+scratch)\n", float64(nbytes)/(1<<30))
	return e, nil
}

func (e *hybridEngine) deriveMaxDims(spec *HybridSpec) {
	for i := range spec.Blocks {
		b := &spec.Blocks[i]
		if b.LayerType == "full_attention" {
			qDim := b.NumHeads * b.HeadDim
			kvDim := b.NumKVHeads * b.HeadDim
			qGate := qDim
			if b.OutputGate {
				qGate = qDim * 2
			}
			if qDim > e.maxQDim {
				e.maxQDim = qDim
			}
			if kvDim > e.maxKVDim {
				e.maxKVDim = kvDim
			}
			if qGate > e.maxQGate {
				e.maxQGate = qGate
			}
		}
		if b.LayerType == "linear_attention" {
			keyDim := b.NumKeyHeads * b.KeyHeadDim
			valDim := b.NumValueHeads * b.ValueHeadDim
			convDim := keyDim*2 + valDim
			k := b.ConvKernel
			if k < 1 {
				k = 1
			}
			hist := k - 1
			if convDim > e.maxConvDim {
				e.maxConvDim = convDim
			}
			if valDim > e.maxValDim {
				e.maxValDim = valDim
			}
			if b.NumValueHeads > e.maxNumV {
				e.maxNumV = b.NumValueHeads
			}
			if b.NumKeyHeads > e.maxNumK {
				e.maxNumK = b.NumKeyHeads
			}
			if b.KeyHeadDim > e.maxHdK {
				e.maxHdK = b.KeyHeadDim
			}
			if b.ValueHeadDim > e.maxHdV {
				e.maxHdV = b.ValueHeadDim
			}
			if hist > e.maxConvHist {
				e.maxConvHist = hist
			}
		}
	}
}

func (e *hybridEngine) estimateVRAM() uint64 {
	var n uint64
	for _, b := range e.owned {
		if b != nil {
			n += b.GetSize()
		}
	}
	return n
}

func (e *hybridEngine) createPipeline(wgsl string) (*wgpu.ComputePipeline, error) {
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

func (e *hybridEngine) mkBuf(label string, size uint64, usage wgpu.BufferUsage, data []byte) (*wgpu.Buffer, error) {
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
		return nil, fmt.Errorf("CreateBuffer %s (%d bytes): %w", label, size, err)
	}
	e.owned = append(e.owned, b)
	return b, nil
}

func (e *hybridEngine) uploadBin(label string, s BinarySpec) (binGPU, error) {
	if s.Rows <= 0 || s.Cols <= 0 || len(s.Words) == 0 {
		return binGPU{}, fmt.Errorf("%s: empty binary matrix", label)
	}
	sc, err := e.mkBuf(label+"_s", uint64(len(s.Scales)*4), wgpu.BufferUsageStorage, f32Bytes(s.Scales))
	if err != nil {
		return binGPU{}, err
	}
	w, err := e.mkBuf(label+"_w", uint64(len(s.Words)*4), wgpu.BufferUsageStorage, u32Bytes(s.Words))
	if err != nil {
		return binGPU{}, err
	}
	return binGPU{scales: sc, weights: w, rows: s.Rows, cols: s.Cols}, nil
}

func onesF32Hybrid(n int) []float32 {
	o := make([]float32, n)
	for i := range o {
		o[i] = 1
	}
	return o
}

func (e *hybridEngine) uploadAll(spec *HybridSpec) error {
	var err error
	if e.embed, err = e.uploadBin("embed", spec.Embed); err != nil {
		return err
	}
	if e.lmHead, err = e.uploadBin("lm", spec.LMHead); err != nil {
		return err
	}
	fn := spec.FinalNorm
	if len(fn) == 0 {
		fn = onesF32Hybrid(spec.Hidden)
	}
	if e.finalNorm, err = e.mkBuf("fnorm", uint64(len(fn)*4), wgpu.BufferUsageStorage, f32Bytes(fn)); err != nil {
		return err
	}

	e.blocks = make([]hybridBlockGPU, spec.Layers)
	for i := range spec.Blocks {
		b := &spec.Blocks[i]
		g := &e.blocks[i]
		g.layerType = b.LayerType
		if g.attnNorm, err = e.mkBuf(fmt.Sprintf("an_%d", i), uint64(len(b.AttnNorm)*4), wgpu.BufferUsageStorage, f32Bytes(b.AttnNorm)); err != nil {
			return err
		}
		if g.ffnNorm, err = e.mkBuf(fmt.Sprintf("fn_%d", i), uint64(len(b.FFNNorm)*4), wgpu.BufferUsageStorage, f32Bytes(b.FFNNorm)); err != nil {
			return err
		}
		if g.gate, err = e.uploadBin(fmt.Sprintf("gate_%d", i), b.Gate); err != nil {
			return err
		}
		if g.up, err = e.uploadBin(fmt.Sprintf("up_%d", i), b.Up); err != nil {
			return err
		}
		if g.down, err = e.uploadBin(fmt.Sprintf("down_%d", i), b.Down); err != nil {
			return err
		}

		switch b.LayerType {
		case "full_attention":
			g.outputGate = b.OutputGate
			g.partialRotary = b.PartialRotary
			g.ropeTheta = b.RoPETheta
			g.numHeads = b.NumHeads
			g.numKVHeads = b.NumKVHeads
			g.headDim = b.HeadDim
			g.qRows = b.Q.Rows
			if g.q, err = e.uploadBin(fmt.Sprintf("q_%d", i), b.Q); err != nil {
				return err
			}
			if g.k, err = e.uploadBin(fmt.Sprintf("k_%d", i), b.K); err != nil {
				return err
			}
			if g.v, err = e.uploadBin(fmt.Sprintf("v_%d", i), b.V); err != nil {
				return err
			}
			if g.o, err = e.uploadBin(fmt.Sprintf("o_%d", i), b.O); err != nil {
				return err
			}
			if g.qNorm, err = e.mkBuf(fmt.Sprintf("qn_%d", i), uint64(len(b.QNorm)*4), wgpu.BufferUsageStorage, f32Bytes(b.QNorm)); err != nil {
				return err
			}
			if g.kNorm, err = e.mkBuf(fmt.Sprintf("kn_%d", i), uint64(len(b.KNorm)*4), wgpu.BufferUsageStorage, f32Bytes(b.KNorm)); err != nil {
				return err
			}
			kvBytes := uint64(b.NumKVHeads * e.maxSeq * b.HeadDim * 4)
			if g.kCache, err = e.mkBuf(fmt.Sprintf("kc_%d", i), kvBytes, wgpu.BufferUsageStorage, nil); err != nil {
				return err
			}
			if g.vCache, err = e.mkBuf(fmt.Sprintf("vc_%d", i), kvBytes, wgpu.BufferUsageStorage, nil); err != nil {
				return err
			}
		case "linear_attention":
			g.numKeyHeads = b.NumKeyHeads
			g.numValueHeads = b.NumValueHeads
			g.keyHeadDim = b.KeyHeadDim
			g.valueHeadDim = b.ValueHeadDim
			g.convKernel = b.ConvKernel
			if g.convKernel < 1 {
				g.convKernel = 1
			}
			if g.gdnQKV, err = e.uploadBin(fmt.Sprintf("gqkv_%d", i), b.GDNQKV); err != nil {
				return err
			}
			if g.gdnZ, err = e.uploadBin(fmt.Sprintf("gz_%d", i), b.GDNZ); err != nil {
				return err
			}
			if g.gdnB, err = e.uploadBin(fmt.Sprintf("gb_%d", i), b.GDNB); err != nil {
				return err
			}
			if g.gdnA, err = e.uploadBin(fmt.Sprintf("ga_%d", i), b.GDNA); err != nil {
				return err
			}
			if g.gdnOut, err = e.uploadBin(fmt.Sprintf("gout_%d", i), b.GDNOut); err != nil {
				return err
			}
			if g.gdnConv, err = e.mkBuf(fmt.Sprintf("gc_%d", i), uint64(len(b.GDNConv)*4), wgpu.BufferUsageStorage, f32Bytes(b.GDNConv)); err != nil {
				return err
			}
			if g.gdnALog, err = e.mkBuf(fmt.Sprintf("gal_%d", i), uint64(len(b.GDNALog)*4), wgpu.BufferUsageStorage, f32Bytes(b.GDNALog)); err != nil {
				return err
			}
			if g.gdnDtBias, err = e.mkBuf(fmt.Sprintf("gdt_%d", i), uint64(len(b.GDNDtBias)*4), wgpu.BufferUsageStorage, f32Bytes(b.GDNDtBias)); err != nil {
				return err
			}
			if g.gdnNorm, err = e.mkBuf(fmt.Sprintf("gn_%d", i), uint64(len(b.GDNNorm)*4), wgpu.BufferUsageStorage, f32Bytes(b.GDNNorm)); err != nil {
				return err
			}
			stBytes := uint64(b.NumValueHeads * b.KeyHeadDim * b.ValueHeadDim * 4)
			if g.gdnState, err = e.mkBuf(fmt.Sprintf("gst_%d", i), stBytes, wgpu.BufferUsageStorage, nil); err != nil {
				return err
			}
			keyDim := b.NumKeyHeads * b.KeyHeadDim
			valDim := b.NumValueHeads * b.ValueHeadDim
			convDim := keyDim*2 + valDim
			hist := g.convKernel - 1
			csBytes := uint64(convDim * hist * 4)
			if csBytes < 64 {
				csBytes = 64
			}
			if g.gdnConvState, err = e.mkBuf(fmt.Sprintf("gcs_%d", i), csBytes, wgpu.BufferUsageStorage, nil); err != nil {
				return err
			}
		default:
			return fmt.Errorf("hybrid layer %d: unknown type %q", i, b.LayerType)
		}
		if (i+1)%8 == 0 || i+1 == spec.Layers {
			fmt.Printf("  hybrid fuse upload layers %d/%d\n", i+1, spec.Layers)
		}
	}
	return nil
}

func (e *hybridEngine) allocScratch() error {
	H := uint64(e.hiddenN * 4)
	var err error
	if e.step, err = e.mkBuf("step", 64, wgpu.BufferUsageStorage, nil); err != nil {
		return err
	}
	if e.token, err = e.mkBuf("token", 64, wgpu.BufferUsageStorage, nil); err != nil {
		return err
	}
	if e.hidden, err = e.mkBuf("h", H, wgpu.BufferUsageStorage, nil); err != nil {
		return err
	}
	if e.normed, err = e.mkBuf("norm", H, wgpu.BufferUsageStorage, nil); err != nil {
		return err
	}
	if e.mix, err = e.mkBuf("mix", H, wgpu.BufferUsageStorage, nil); err != nil {
		return err
	}
	if e.inter, err = e.mkBuf("inter", uint64(e.interN*4), wgpu.BufferUsageStorage, nil); err != nil {
		return err
	}
	if e.upBuf, err = e.mkBuf("upbuf", uint64(e.interN*4), wgpu.BufferUsageStorage, nil); err != nil {
		return err
	}
	if e.logits, err = e.mkBuf("logits", uint64(e.vocabN*4), wgpu.BufferUsageStorage, nil); err != nil {
		return err
	}
	if e.stagingLogits, err = e.mkBuf("stageLogits", uint64(e.vocabN*4), wgpu.BufferUsageMapRead, nil); err != nil {
		return err
	}

	if e.maxQGate > 0 {
		if e.qGate, err = e.mkBuf("qgate", uint64(e.maxQGate*4), wgpu.BufferUsageStorage, nil); err != nil {
			return err
		}
		if e.qBuf, err = e.mkBuf("q", uint64(e.maxQDim*4), wgpu.BufferUsageStorage, nil); err != nil {
			return err
		}
		if e.gateBuf, err = e.mkBuf("gate", uint64(e.maxQDim*4), wgpu.BufferUsageStorage, nil); err != nil {
			return err
		}
		if e.kBuf, err = e.mkBuf("k", uint64(e.maxKVDim*4), wgpu.BufferUsageStorage, nil); err != nil {
			return err
		}
		if e.vBuf, err = e.mkBuf("v", uint64(e.maxKVDim*4), wgpu.BufferUsageStorage, nil); err != nil {
			return err
		}
		if e.attnOut, err = e.mkBuf("ao", uint64(e.maxQDim*4), wgpu.BufferUsageStorage, nil); err != nil {
			return err
		}
	}
	if e.maxConvDim > 0 {
		if e.gdnQKV, err = e.mkBuf("gqkv", uint64(e.maxConvDim*4), wgpu.BufferUsageStorage, nil); err != nil {
			return err
		}
		if e.gdnZ, err = e.mkBuf("gz", uint64(e.maxValDim*4), wgpu.BufferUsageStorage, nil); err != nil {
			return err
		}
		if e.gdnBetaRaw, err = e.mkBuf("gbr", uint64(e.maxNumV*4), wgpu.BufferUsageStorage, nil); err != nil {
			return err
		}
		if e.gdnARaw, err = e.mkBuf("gar", uint64(e.maxNumV*4), wgpu.BufferUsageStorage, nil); err != nil {
			return err
		}
		if e.gdnMixed, err = e.mkBuf("gmix", uint64(e.maxConvDim*4), wgpu.BufferUsageStorage, nil); err != nil {
			return err
		}
		qRepN := e.maxNumV * e.maxHdK
		if e.gdnQRep, err = e.mkBuf("gqrep", uint64(qRepN*4), wgpu.BufferUsageStorage, nil); err != nil {
			return err
		}
		if e.gdnKRep, err = e.mkBuf("gkrep", uint64(qRepN*4), wgpu.BufferUsageStorage, nil); err != nil {
			return err
		}
		if e.gdnBeta, err = e.mkBuf("gbeta", uint64(e.maxNumV*4), wgpu.BufferUsageStorage, nil); err != nil {
			return err
		}
		if e.gdnG, err = e.mkBuf("gg", uint64(e.maxNumV*4), wgpu.BufferUsageStorage, nil); err != nil {
			return err
		}
		if e.gdnCore, err = e.mkBuf("gcore", uint64(e.maxValDim*4), wgpu.BufferUsageStorage, nil); err != nil {
			return err
		}
	}
	return nil
}

func (e *hybridEngine) uni(label string, bytes []byte) *wgpu.Buffer {
	b, err := e.mkBuf(label, 256, wgpu.BufferUsageUniform, bytes)
	if err != nil {
		panic(err)
	}
	return b
}

func (e *hybridEngine) initUniforms() {
	e.uRMS = e.uni("uRMS", packMix(uint32(e.hiddenN), e.eps, 0, 0))
	e.uResidH = e.uni("uRH", packU32(uint32(e.hiddenN), 0, 0, 0))
	e.uSwiglu = e.uni("uSW", packU32(uint32(e.interN), 0, 0, 0))
	e.uEmbed = e.uni("uEM", packU32(uint32(e.hiddenN), uint32(e.embed.cols/32), uint32(e.embed.cols/128), 0))
	e.uGemvVocabH = e.uni("uVH", packU32(uint32(e.hiddenN), uint32(e.vocabN), 0, 0))
	e.uGemvHInter = e.uni("uDown", packU32(uint32(e.interN), uint32(e.hiddenN), 0, 0))
	e.uGemvInterH = e.uni("uGate", packU32(uint32(e.hiddenN), uint32(e.interN), 0, 0))
	e.uZero = e.uni("uZero", packU32(0, 0, 0, 0))
}

func (e *hybridEngine) mkBG(key string, pipe *wgpu.ComputePipeline, slices ...bufSlice) *wgpu.BindGroup {
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
		panic(fmt.Sprintf("bindgroup %s: %v", key, err))
	}
	e.bg[key] = bg
	return bg
}

func (e *hybridEngine) gemvU(label string, cols, rows int) *wgpu.Buffer {
	return e.uni(label, packU32(uint32(cols), uint32(rows), 0, 0))
}

func (e *hybridEngine) buildBindGroups() {
	p := e.pipe
	e.mkBG("embed", p["binembed"], whole(e.uEmbed), whole(e.token), whole(e.embed.scales), whole(e.embed.weights), whole(e.hidden))
	e.mkBG("fnorm", p["rmsnorm"], whole(e.uRMS), whole(e.hidden), whole(e.finalNorm), whole(e.normed))
	e.mkBG("lm", p["bingemv"], whole(e.uGemvVocabH), whole(e.normed), whole(e.lmHead.scales), whole(e.lmHead.weights), whole(e.logits))
	e.mkBG("inc_pos", p["inc_pos"], whole(e.step))

	for i := range e.blocks {
		b := &e.blocks[i]
		tag := fmt.Sprintf("L%d", i)
		e.mkBG(tag+"_rms1", p["rmsnorm"], whole(e.uRMS), whole(e.hidden), whole(b.attnNorm), whole(e.normed))
		e.mkBG(tag+"_rms2", p["rmsnorm"], whole(e.uRMS), whole(e.hidden), whole(b.ffnNorm), whole(e.normed))
		uGate := e.gemvU(tag+"_uGate", b.gate.cols, b.gate.rows)
		uUp := e.gemvU(tag+"_uUp", b.up.cols, b.up.rows)
		uDown := e.gemvU(tag+"_uDown", b.down.cols, b.down.rows)
		e.mkBG(tag+"_gate", p["bingemv"], whole(uGate), whole(e.normed), whole(b.gate.scales), whole(b.gate.weights), whole(e.inter))
		e.mkBG(tag+"_up", p["bingemv"], whole(uUp), whole(e.normed), whole(b.up.scales), whole(b.up.weights), whole(e.upBuf))
		e.mkBG(tag+"_sw", p["swiglu"], whole(e.uSwiglu), whole(e.inter), whole(e.upBuf))
		e.mkBG(tag+"_down", p["bingemv"], whole(uDown), whole(e.inter), whole(b.down.scales), whole(b.down.weights), whole(e.mix))
		e.mkBG(tag+"_r1", p["resid"], whole(e.uResidH), whole(e.hidden), whole(e.mix))
		e.mkBG(tag+"_r2", p["resid"], whole(e.uResidH), whole(e.hidden), whole(e.mix))

		switch b.layerType {
		case "full_attention":
			e.buildAttnBGs(tag, b)
		case "linear_attention":
			e.buildGDNBGs(tag, b)
		}
	}
}

func (e *hybridEngine) buildAttnBGs(tag string, b *hybridBlockGPU) {
	p := e.pipe
	qDim := b.numHeads * b.headDim
	kvDim := b.numKVHeads * b.headDim
	uQ := e.gemvU(tag+"_uQ", b.q.cols, b.q.rows)
	uK := e.gemvU(tag+"_uK", b.k.cols, b.k.rows)
	uV := e.gemvU(tag+"_uV", b.v.cols, b.v.rows)
	uO := e.gemvU(tag+"_uO", b.o.cols, b.o.rows)

	qOut := e.qBuf
	if b.outputGate {
		qOut = e.qGate
	}
	e.mkBG(tag+"_q", p["bingemv"], whole(uQ), whole(e.normed), whole(b.q.scales), whole(b.q.weights), whole(qOut))
	e.mkBG(tag+"_k", p["bingemv"], whole(uK), whole(e.normed), whole(b.k.scales), whole(b.k.weights), whole(e.kBuf))
	e.mkBG(tag+"_v", p["bingemv"], whole(uV), whole(e.normed), whole(b.v.scales), whole(b.v.weights), whole(e.vBuf))

	if b.outputGate {
		uSplit := e.uni(tag+"_uSplit", packU32(uint32(b.numHeads), uint32(b.headDim), 0, 0))
		e.mkBG(tag+"_split", p["split_qg"], whole(uSplit), whole(e.qGate), whole(e.qBuf), whole(e.gateBuf))
	}
	uHQ := e.uni(tag+"_uHQ", packU32(uint32(b.numHeads), uint32(b.headDim), mathFloat32bits(e.eps), 0))
	uHK := e.uni(tag+"_uHK", packU32(uint32(b.numKVHeads), uint32(b.headDim), mathFloat32bits(e.eps), 0))
	e.mkBG(tag+"_hrmsq", p["head_rms"], whole(uHQ), whole(e.qBuf), whole(b.qNorm))
	e.mkBG(tag+"_hrmsk", p["head_rms"], whole(uHK), whole(e.kBuf), whole(b.kNorm))

	rotDim := int(float64(b.headDim) * float64(b.partialRotary))
	if rotDim <= 0 {
		rotDim = b.headDim
	}
	if rotDim%2 != 0 {
		rotDim--
	}
	theta := b.ropeTheta
	if theta <= 0 {
		theta = 10000
	}
	uRQ := e.uni(tag+"_uRQ", packU32(uint32(b.numHeads), uint32(b.headDim), uint32(rotDim), mathFloat32bits(theta)))
	uRK := e.uni(tag+"_uRK", packU32(uint32(b.numKVHeads), uint32(b.headDim), uint32(rotDim), mathFloat32bits(theta)))
	e.mkBG(tag+"_ropeq", p["prope"], whole(uRQ), whole(e.step), whole(e.qBuf))
	e.mkBG(tag+"_ropek", p["prope"], whole(uRK), whole(e.step), whole(e.kBuf))

	uKV := e.uni(tag+"_uKV", packU32(uint32(kvDim), uint32(e.maxSeq), uint32(b.headDim), 0))
	e.mkBG(tag+"_kv", p["kv"], whole(uKV), whole(e.step), whole(e.kBuf), whole(e.vBuf), whole(b.kCache), whole(b.vCache))
	uAttn := e.uni(tag+"_uAttn", packU32(uint32(b.numHeads), uint32(b.numKVHeads), uint32(b.headDim), uint32(e.maxSeq)))
	e.mkBG(tag+"_attn", p["attn"], whole(uAttn), whole(e.step), whole(e.qBuf), whole(b.kCache), whole(b.vCache), whole(e.attnOut))

	if b.outputGate {
		uOG := e.uni(tag+"_uOG", packU32(uint32(qDim), 0, 0, 0))
		e.mkBG(tag+"_ogate", p["outgate"], whole(uOG), whole(e.attnOut), whole(e.gateBuf))
	}
	e.mkBG(tag+"_o", p["bingemv"], whole(uO), whole(e.attnOut), whole(b.o.scales), whole(b.o.weights), whole(e.mix))
}

func (e *hybridEngine) buildGDNBGs(tag string, b *hybridBlockGPU) {
	p := e.pipe
	keyDim := b.numKeyHeads * b.keyHeadDim
	valDim := b.numValueHeads * b.valueHeadDim
	convDim := keyDim*2 + valDim

	uQKV := e.gemvU(tag+"_uQKV", b.gdnQKV.cols, b.gdnQKV.rows)
	uZ := e.gemvU(tag+"_uZ", b.gdnZ.cols, b.gdnZ.rows)
	uB := e.gemvU(tag+"_uB", b.gdnB.cols, b.gdnB.rows)
	uA := e.gemvU(tag+"_uA", b.gdnA.cols, b.gdnA.rows)
	uOut := e.gemvU(tag+"_uOut", b.gdnOut.cols, b.gdnOut.rows)

	e.mkBG(tag+"_gqkv", p["bingemv"], whole(uQKV), whole(e.normed), whole(b.gdnQKV.scales), whole(b.gdnQKV.weights), whole(e.gdnQKV))
	e.mkBG(tag+"_gz", p["bingemv"], whole(uZ), whole(e.normed), whole(b.gdnZ.scales), whole(b.gdnZ.weights), whole(e.gdnZ))
	e.mkBG(tag+"_gb", p["bingemv"], whole(uB), whole(e.normed), whole(b.gdnB.scales), whole(b.gdnB.weights), whole(e.gdnBetaRaw))
	e.mkBG(tag+"_ga", p["bingemv"], whole(uA), whole(e.normed), whole(b.gdnA.scales), whole(b.gdnA.weights), whole(e.gdnARaw))

	uConv := e.uni(tag+"_uConv", packU32(uint32(convDim), uint32(b.convKernel), 0, 0))
	e.mkBG(tag+"_gconv", p["gdn_conv"], whole(uConv), whole(e.gdnQKV), whole(b.gdnConv), whole(b.gdnConvState), whole(e.gdnMixed))

	uPrep := e.uni(tag+"_uPrep", packU32(uint32(b.numKeyHeads), uint32(b.numValueHeads), uint32(b.keyHeadDim), uint32(b.valueHeadDim)))
	e.mkBG(tag+"_gprep", p["gdn_prepqk"], whole(uPrep), whole(e.gdnMixed), whole(e.gdnQRep), whole(e.gdnKRep))

	uBG := e.uni(tag+"_uBG", packU32(uint32(b.numValueHeads), 0, 0, 0))
	e.mkBG(tag+"_gbetag", p["gdn_betag"], whole(uBG), whole(e.gdnBetaRaw), whole(e.gdnARaw), whole(b.gdnALog), whole(b.gdnDtBias), whole(e.gdnBeta), whole(e.gdnG))

	uStep := e.uni(tag+"_uStep", packU32(uint32(b.numValueHeads), uint32(b.numKeyHeads), uint32(b.keyHeadDim), uint32(b.valueHeadDim)))
	vOff := uint64(keyDim * 2 * 4)
	vSize := uint64(valDim * 4)
	e.mkBG(tag+"_gstep", p["gdn_step"], whole(uStep), whole(e.gdnQRep), whole(e.gdnKRep),
		bufSlice{e.gdnMixed, vOff, vSize}, whole(e.gdnBeta), whole(e.gdnG), whole(b.gdnState), whole(e.gdnCore))

	uGN := e.uni(tag+"_uGN", packMix(uint32(b.valueHeadDim), e.eps, 0, 0))
	e.mkBG(tag+"_ggnorm", p["gdn_gnorm"], whole(uGN), whole(e.gdnCore), whole(e.gdnZ), whole(b.gdnNorm))
	e.mkBG(tag+"_gout", p["bingemv"], whole(uOut), whole(e.gdnCore), whole(b.gdnOut.scales), whole(b.gdnOut.weights), whole(e.mix))
}

func (e *hybridEngine) disp(pass *wgpu.ComputePassEncoder, pipe *wgpu.ComputePipeline, bg *wgpu.BindGroup, x, y, z uint32) {
	pass.SetPipeline(pipe)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(x, y, z)
}

func (e *hybridEngine) recordLayers(pass *wgpu.ComputePassEncoder) {
	p := e.pipe
	hWG := (uint32(e.hiddenN) + 63) / 64
	iWG := (uint32(e.interN) + 63) / 64
	for i := range e.blocks {
		b := &e.blocks[i]
		tag := fmt.Sprintf("L%d", i)
		e.disp(pass, p["rmsnorm"], e.bg[tag+"_rms1"], 1, 1, 1)

		switch b.layerType {
		case "full_attention":
			qDim := b.numHeads * b.headDim
			kvDim := b.numKVHeads * b.headDim
			e.disp(pass, p["bingemv"], e.bg[tag+"_q"], (uint32(b.q.rows)+63)/64, 1, 1)
			e.disp(pass, p["bingemv"], e.bg[tag+"_k"], (uint32(b.k.rows)+63)/64, 1, 1)
			e.disp(pass, p["bingemv"], e.bg[tag+"_v"], (uint32(b.v.rows)+63)/64, 1, 1)
			if b.outputGate {
				e.disp(pass, p["split_qg"], e.bg[tag+"_split"], (uint32(qDim)+63)/64, 1, 1)
			}
			e.disp(pass, p["head_rms"], e.bg[tag+"_hrmsq"], uint32(b.numHeads), 1, 1)
			e.disp(pass, p["head_rms"], e.bg[tag+"_hrmsk"], uint32(b.numKVHeads), 1, 1)
			e.disp(pass, p["prope"], e.bg[tag+"_ropeq"], (uint32(b.numHeads)+63)/64, 1, 1)
			e.disp(pass, p["prope"], e.bg[tag+"_ropek"], (uint32(b.numKVHeads)+63)/64, 1, 1)
			e.disp(pass, p["kv"], e.bg[tag+"_kv"], (uint32(kvDim)+63)/64, 1, 1)
			e.disp(pass, p["attn"], e.bg[tag+"_attn"], uint32(b.numHeads), 1, 1)
			if b.outputGate {
				e.disp(pass, p["outgate"], e.bg[tag+"_ogate"], (uint32(qDim)+63)/64, 1, 1)
			}
			e.disp(pass, p["bingemv"], e.bg[tag+"_o"], (uint32(b.o.rows)+63)/64, 1, 1)
		case "linear_attention":
			keyDim := b.numKeyHeads * b.keyHeadDim
			convDim := keyDim*2 + b.numValueHeads*b.valueHeadDim
			e.disp(pass, p["bingemv"], e.bg[tag+"_gqkv"], (uint32(b.gdnQKV.rows)+63)/64, 1, 1)
			e.disp(pass, p["bingemv"], e.bg[tag+"_gz"], (uint32(b.gdnZ.rows)+63)/64, 1, 1)
			e.disp(pass, p["bingemv"], e.bg[tag+"_gb"], (uint32(b.gdnB.rows)+63)/64, 1, 1)
			e.disp(pass, p["bingemv"], e.bg[tag+"_ga"], (uint32(b.gdnA.rows)+63)/64, 1, 1)
			e.disp(pass, p["gdn_conv"], e.bg[tag+"_gconv"], (uint32(convDim)+63)/64, 1, 1)
			e.disp(pass, p["gdn_prepqk"], e.bg[tag+"_gprep"], uint32(b.numKeyHeads), 1, 1)
			e.disp(pass, p["gdn_betag"], e.bg[tag+"_gbetag"], (uint32(b.numValueHeads)+63)/64, 1, 1)
			e.disp(pass, p["gdn_step"], e.bg[tag+"_gstep"], uint32(b.numValueHeads), 1, 1)
			e.disp(pass, p["gdn_gnorm"], e.bg[tag+"_ggnorm"], uint32(b.numValueHeads), 1, 1)
			e.disp(pass, p["bingemv"], e.bg[tag+"_gout"], (uint32(b.gdnOut.rows)+63)/64, 1, 1)
		}
		e.disp(pass, p["resid"], e.bg[tag+"_r1"], hWG, 1, 1)

		e.disp(pass, p["rmsnorm"], e.bg[tag+"_rms2"], 1, 1, 1)
		e.disp(pass, p["bingemv"], e.bg[tag+"_gate"], (uint32(b.gate.rows)+63)/64, 1, 1)
		e.disp(pass, p["bingemv"], e.bg[tag+"_up"], (uint32(b.up.rows)+63)/64, 1, 1)
		e.disp(pass, p["swiglu"], e.bg[tag+"_sw"], iWG, 1, 1)
		e.disp(pass, p["bingemv"], e.bg[tag+"_down"], (uint32(b.down.rows)+63)/64, 1, 1)
		e.disp(pass, p["resid"], e.bg[tag+"_r2"], hWG, 1, 1)
	}
	e.disp(pass, p["rmsnorm"], e.bg["fnorm"], 1, 1, 1)
}

func (e *hybridEngine) appendTokens(ids []uint32) ([]float32, error) {
	if len(ids) == 0 {
		return nil, fmt.Errorf("fusedgpu: empty ids")
	}
	logits := make([]float32, e.vocabN)
	for i, id := range ids {
		if err := e.stepToken(id, i == len(ids)-1, logits); err != nil {
			return nil, err
		}
	}
	return logits, nil
}

func (e *hybridEngine) stepToken(id uint32, wantLogits bool, logits []float32) error {
	e.queue.WriteBuffer(e.step, 0, packU32(uint32(e.pos), 0))
	e.queue.WriteBuffer(e.token, 0, packU32(id))

	enc, err := e.device.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	pass := enc.BeginComputePass(nil)
	e.disp(pass, e.pipe["binembed"], e.bg["embed"], (uint32(e.hiddenN)+63)/64, 1, 1)
	e.recordLayers(pass)
	if wantLogits {
		e.disp(pass, e.pipe["bingemv"], e.bg["lm"], (uint32(e.vocabN)+63)/64, 1, 1)
	}
	e.disp(pass, e.pipe["inc_pos"], e.bg["inc_pos"], 1, 1, 1)
	pass.End()

	if wantLogits {
		bytes := uint64(e.vocabN * 4)
		enc.CopyBufferToBuffer(e.logits, 0, e.stagingLogits, 0, bytes)
		cmd, err := enc.Finish(nil)
		if err != nil {
			return err
		}
		e.queue.Submit(cmd)
		if err := e.readLogits(logits); err != nil {
			return err
		}
	} else {
		cmd, err := enc.Finish(nil)
		if err != nil {
			return err
		}
		e.queue.Submit(cmd)
	}
	e.pos++
	return nil
}

func (e *hybridEngine) readLogits(dst []float32) error {
	bytes := uint64(len(dst) * 4)
	done := make(chan struct{})
	var st wgpu.BufferMapAsyncStatus
	if err := e.stagingLogits.MapAsync(wgpu.MapModeRead, 0, bytes, func(status wgpu.BufferMapAsyncStatus) {
		st = status
		close(done)
	}); err != nil {
		return err
	}
	deadline := time.Now().Add(300 * time.Second)
	for {
		e.device.Poll(false, nil)
		select {
		case <-done:
			if st != wgpu.BufferMapAsyncStatusSuccess {
				return fmt.Errorf("fusedgpu hybrid MapAsync %v", st)
			}
			raw := e.stagingLogits.GetMappedRange(0, uint(bytes))
			for i := range dst {
				dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4 : i*4+4]))
			}
			e.stagingLogits.Unmap()
			return nil
		default:
			if time.Now().After(deadline) {
				return fmt.Errorf("fusedgpu hybrid MapAsync timeout")
			}
			runtime.Gosched()
		}
	}
}

func (e *hybridEngine) resetState() error {
	e.pos = 0
	e.queue.WriteBuffer(e.step, 0, packU32(0, 0))

	enc, err := e.device.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	pass := enc.BeginComputePass(nil)
	p := e.pipe
	for i := range e.blocks {
		b := &e.blocks[i]
		tag := fmt.Sprintf("clr%d", i)
		if b.kCache != nil {
			n := uint32(b.numKVHeads * e.maxSeq * b.headDim)
			u := e.uni(tag+"_kc", packU32(n, 0, 0, 0))
			bg := e.mkBG(tag+"_kc", p["zero"], whole(u), whole(b.kCache))
			e.disp(pass, p["zero"], bg, (n+63)/64, 1, 1)
			u2 := e.uni(tag+"_vc", packU32(n, 0, 0, 0))
			bg2 := e.mkBG(tag+"_vc", p["zero"], whole(u2), whole(b.vCache))
			e.disp(pass, p["zero"], bg2, (n+63)/64, 1, 1)
		}
		if b.gdnState != nil {
			n := uint32(b.numValueHeads * b.keyHeadDim * b.valueHeadDim)
			u := e.uni(tag+"_gs", packU32(n, 0, 0, 0))
			bg := e.mkBG(tag+"_gs", p["zero"], whole(u), whole(b.gdnState))
			e.disp(pass, p["zero"], bg, (n+63)/64, 1, 1)
			keyDim := b.numKeyHeads * b.keyHeadDim
			valDim := b.numValueHeads * b.valueHeadDim
			convDim := keyDim*2 + valDim
			hist := b.convKernel - 1
			if hist < 0 {
				hist = 0
			}
			nc := uint32(convDim * hist)
			if nc > 0 {
				u2 := e.uni(tag+"_gcs", packU32(nc, 0, 0, 0))
				bg2 := e.mkBG(tag+"_gcs", p["zero"], whole(u2), whole(b.gdnConvState))
				e.disp(pass, p["zero"], bg2, (nc+63)/64, 1, 1)
			}
		}
	}
	pass.End()
	cmd, err := enc.Finish(nil)
	if err != nil {
		return err
	}
	e.queue.Submit(cmd)
	e.device.Poll(true, nil)
	return nil
}

func (e *hybridEngine) release() {
	if e == nil {
		return
	}
	if e.device != nil {
		e.device.Poll(true, nil)
	}
	for _, bg := range e.bg {
		if bg != nil {
			bg.Release()
		}
	}
	e.bg = nil
	for _, p := range e.pipe {
		if p != nil {
			p.Release()
		}
	}
	e.pipe = nil
	for _, b := range e.owned {
		if b != nil {
			b.Release()
		}
	}
	e.owned = nil
	e.blocks = nil
	e.device, e.queue, e.adapter, e.instance = nil, nil, nil, nil
	e.spec = nil
	runtime.GC()
}

var _ = unsafe.Sizeof(0)
