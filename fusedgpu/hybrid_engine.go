package fusedgpu

import (
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"runtime/debug"
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
	promptBuf           *wgpu.Buffer
	histBuf             *wgpu.Buffer
	stagingHist         *wgpu.Buffer
	hidden, normed, mix *wgpu.Buffer
	inter, upBuf        *wgpu.Buffer
	logits              *wgpu.Buffer
	stagingLogits       *wgpu.Buffer
	outTok              *wgpu.Buffer
	stagingTok          *wgpu.Buffer

	qGate, qBuf, gateBuf, kBuf, vBuf, attnOut *wgpu.Buffer

	gdnQKV, gdnZ, gdnBetaRaw, gdnARaw *wgpu.Buffer
	gdnMixed, gdnQRep, gdnKRep        *wgpu.Buffer
	gdnBeta, gdnG, gdnCore            *wgpu.Buffer

	uRMS, uResidH, uSwiglu   *wgpu.Buffer
	uEmbed                   *wgpu.Buffer
	uGemvVocabH              *wgpu.Buffer
	uGemvHInter, uGemvInterH *wgpu.Buffer
	uArgMax                  *wgpu.Buffer
	uZero                    *wgpu.Buffer

	hiddenN, vocabN, interN, maxSeq int
	eps                             float32
	pos                             int
	lmShards                        []lmShard

	maxQDim, maxKVDim, maxQGate int
	maxConvDim, maxValDim       int
	maxNumV, maxHdK, maxHdV     int
	maxNumK                     int
	maxConvHist                 int

	dispCount int
	useSPIRV  bool
}

type lmShard struct {
	bg        *wgpu.BindGroup
	workgroups uint32
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
		e.maxSeq = DefaultMaxSeq
	}
	e.maxSeq = ClampAttnMaxSeq(e.maxSeq)
	if e.eps <= 0 {
		e.eps = 1e-6
	}
	e.deriveMaxDims(spec)
	if e.maxHdV > 512 {
		return nil, fmt.Errorf("fusedgpu: value head dim %d > 512 (GDN scratch limit)", e.maxHdV)
	}

	inst, adapt, device, queue, _, err := acquireDevice()
	if err != nil {
		return nil, err
	}
	e.instance = inst
	e.adapter = adapt
	e.device = device
	e.queue = queue
	e.clampMaxSeqForSSBO()

	shaders := hybridShaderWGSL()
	spirvHits := 0
	for name, src := range shaders {
		p, via, err := e.createPipelineNamed(name, src)
		if err != nil {
			e.release()
			return nil, fmt.Errorf("pipeline %s: %w", name, err)
		}
		if via == "spirv" {
			spirvHits++
		}
		e.pipe[name] = p
	}
	e.useSPIRV = spirvHits > 0

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
	fmt.Printf("✅ Hybrid GPU fuse ready (wave1 FA fuse, spirv=%d/%d, ~%.1f GiB)\n",
		spirvHits, len(shaders), float64(nbytes)/(1<<30))
	return e, nil
}

// HybridShaderWGSLExport exposes pipeline WGSL for AOT tooling (dumpwgsl / naga).
func HybridShaderWGSLExport() map[string]string {
	return hybridShaderWGSL()
}

// hybridShaderWGSL is the HybridEngine pipeline source map (WGSL). Used for
// runtime compile and by scripts/compile-hybrid-spirv.sh for AOT SPIR-V.
func hybridShaderWGSL() map[string]string {
	return map[string]string{
		"bingemv":         shaderBinG128GEMV,
		"bingemv_add":     shaderBinG128GEMVAdd,
		"bingemv_dual":    shaderBinG128Dual,
		"bingemv_qkv_rms": shaderBinG128QKVRMS,
		"binswiglu":       shaderBinG128SwiGLU,
		"binswiglu_rms":   shaderBinG128SwiGLURMS,
		"binembed":        shaderBinEmbed,
		"binembed_p":      shaderBinEmbedPrompt,
		"rmsnorm":         shaderHybridRMS,
		"gdn_conv":        shaderGDNConv,
		"gdn_prep":        shaderGDNPrepFused,
		"gdn_step":        shaderGDNStepGNorm,
		"attn_q_prep":     shaderAttnQPrep,
		"attn_kv_prep":    shaderAttnKVPrep,
		"attn_prep":       shaderAttnPrep,
		"attn_gated":      shaderAttnGated,
		"inc_pos":         shaderIncPosHybrid,
		"advance":         shaderAdvance,
		"zero":            shaderZeroF32,
		"argmax":          shaderHybridArgMax,
	}
}

func (e *hybridEngine) clampMaxSeqForSSBO() {
	lim := MaxStorageBindingLimit()
	if lim == 0 || e.maxKVDim <= 0 {
		return
	}
	// Each full-attention block binds K and V cache separately:
	// size = maxKVDim * maxSeq * 4 bytes each, and each binding must fit maxSSBO.
	maxSeqByKV := int(lim / (uint64(e.maxKVDim) * 4))
	if maxSeqByKV <= 0 {
		maxSeqByKV = 1
	}
	maxSeqByKV = ClampAttnMaxSeq(maxSeqByKV)
	if maxSeqByKV < 256 {
		maxSeqByKV = 256
	}
	if e.maxSeq > maxSeqByKV {
		fmt.Printf("  clamp maxSeq for maxSSBO=%d: %d -> %d (maxKVDim=%d)\n",
			lim, e.maxSeq, maxSeqByKV, e.maxKVDim)
		e.maxSeq = maxSeqByKV
	}
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
	p, _, err := e.createPipelineNamed("", wgsl)
	return p, err
}

func (e *hybridEngine) createPipelineNamed(name, wgsl string) (*wgpu.ComputePipeline, string, error) {
	if name != "" {
		if spv, ok := hybridSPIRV[name]; ok && len(spv) >= 4 {
			mod, err := e.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
				Label:           name,
				SPIRVDescriptor: &wgpu.ShaderModuleSPIRVDescriptor{Code: spv},
			})
			if err == nil {
				defer mod.Release()
				p, perr := e.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
					Compute: wgpu.ProgrammableStageDescriptor{Module: mod, EntryPoint: "main"},
				})
				if perr == nil {
					return p, "spirv", nil
				}
				err = perr
			}
			fmt.Printf("  spirv %s fallback to WGSL: %v\n", name, err)
		}
	}
	mod, err := e.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: wgsl},
	})
	if err != nil {
		return nil, "", err
	}
	defer mod.Release()
	p, err := e.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Compute: wgpu.ProgrammableStageDescriptor{Module: mod, EntryPoint: "main"},
	})
	if err != nil {
		return nil, "", err
	}
	return p, "wgsl", nil
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
	if usage&wgpu.BufferUsageStorage != 0 {
		if lim := MaxStorageBindingLimit(); lim > 0 && size > lim {
			return nil, fmt.Errorf(
				"storage buffer %s is %d bytes > adapter maxSSBO %d bytes",
				label, size, lim,
			)
		}
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

func (e *hybridEngine) uploadBin(label string, s *BinarySpec) (binGPU, error) {
	if s == nil || s.Rows <= 0 || s.Cols <= 0 || len(s.Words) == 0 {
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
	// Move to GPU: drop host staging immediately so peak ≈ remaining+device, not 2×.
	s.Scales, s.Words = nil, nil
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
	if e.embed, err = e.uploadBin("embed", &spec.Embed); err != nil {
		return err
	}
	if spec.LMHeadTied {
		// Same GPU buffers as embed — avoid a second host→device copy of the vocab table.
		e.lmHead = e.embed
	} else if e.lmHead, err = e.uploadBin("lm", &spec.LMHead); err != nil {
		return err
	}
	fn := spec.FinalNorm
	if len(fn) == 0 {
		fn = onesF32Hybrid(spec.Hidden)
	}
	if e.finalNorm, err = e.mkBuf("fnorm", uint64(len(fn)*4), wgpu.BufferUsageStorage, f32Bytes(fn)); err != nil {
		return err
	}
	spec.FinalNorm = nil

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
		b.AttnNorm, b.FFNNorm = nil, nil
		if g.gate, err = e.uploadBin(fmt.Sprintf("gate_%d", i), &b.Gate); err != nil {
			return err
		}
		if g.up, err = e.uploadBin(fmt.Sprintf("up_%d", i), &b.Up); err != nil {
			return err
		}
		if g.down, err = e.uploadBin(fmt.Sprintf("down_%d", i), &b.Down); err != nil {
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
			if g.q, err = e.uploadBin(fmt.Sprintf("q_%d", i), &b.Q); err != nil {
				return err
			}
			if g.k, err = e.uploadBin(fmt.Sprintf("k_%d", i), &b.K); err != nil {
				return err
			}
			if g.v, err = e.uploadBin(fmt.Sprintf("v_%d", i), &b.V); err != nil {
				return err
			}
			if g.o, err = e.uploadBin(fmt.Sprintf("o_%d", i), &b.O); err != nil {
				return err
			}
			if g.qNorm, err = e.mkBuf(fmt.Sprintf("qn_%d", i), uint64(len(b.QNorm)*4), wgpu.BufferUsageStorage, f32Bytes(b.QNorm)); err != nil {
				return err
			}
			if g.kNorm, err = e.mkBuf(fmt.Sprintf("kn_%d", i), uint64(len(b.KNorm)*4), wgpu.BufferUsageStorage, f32Bytes(b.KNorm)); err != nil {
				return err
			}
			b.QNorm, b.KNorm = nil, nil
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
			if g.gdnQKV, err = e.uploadBin(fmt.Sprintf("gqkv_%d", i), &b.GDNQKV); err != nil {
				return err
			}
			if g.gdnZ, err = e.uploadBin(fmt.Sprintf("gz_%d", i), &b.GDNZ); err != nil {
				return err
			}
			if g.gdnB, err = e.uploadBin(fmt.Sprintf("gb_%d", i), &b.GDNB); err != nil {
				return err
			}
			if g.gdnA, err = e.uploadBin(fmt.Sprintf("ga_%d", i), &b.GDNA); err != nil {
				return err
			}
			if g.gdnOut, err = e.uploadBin(fmt.Sprintf("gout_%d", i), &b.GDNOut); err != nil {
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
			b.GDNConv, b.GDNALog, b.GDNDtBias, b.GDNNorm = nil, nil, nil, nil
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
			// Reclaim staging pages between layer batches (iPad Jetsam).
			runtime.GC()
			debug.FreeOSMemory()
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
	if e.promptBuf, err = e.mkBuf("prompt", uint64(e.maxSeq*4), wgpu.BufferUsageStorage, nil); err != nil {
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
	if e.outTok, err = e.mkBuf("outTok", 64, wgpu.BufferUsageStorage, nil); err != nil {
		return err
	}
	if e.stagingTok, err = e.mkBuf("stageTok", 64, wgpu.BufferUsageMapRead, nil); err != nil {
		return err
	}
	if e.histBuf, err = e.mkBuf("hist", uint64(e.maxSeq*4), wgpu.BufferUsageStorage, nil); err != nil {
		return err
	}
	if e.stagingHist, err = e.mkBuf("stageHist", uint64(e.maxSeq*4), wgpu.BufferUsageMapRead, nil); err != nil {
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
	// binswiglu: inputSize=hidden, intermediate
	e.uSwiglu = e.uni("uSW", packU32(uint32(e.hiddenN), uint32(e.interN), 0, 0))
	e.uEmbed = e.uni("uEM", packU32(uint32(e.hiddenN), uint32(e.embed.cols/32), uint32(e.embed.cols/128), 0))
	e.uGemvVocabH = e.uni("uVH", packU32(uint32(e.hiddenN), uint32(e.vocabN), 0, 0))
	e.uGemvHInter = e.uni("uDown", packU32(uint32(e.interN), uint32(e.hiddenN), 0, 0))
	e.uGemvInterH = e.uni("uGate", packU32(uint32(e.hiddenN), uint32(e.interN), 0, 0))
	e.uArgMax = e.uni("uAM", packU32(uint32(e.vocabN), 0, 0, 0))
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
	e.mkBG("embed_p", p["binembed_p"], whole(e.uEmbed), whole(e.step), whole(e.promptBuf), whole(e.embed.scales), whole(e.embed.weights), whole(e.hidden))
	e.mkBG("fnorm", p["rmsnorm"], whole(e.uRMS), whole(e.hidden), whole(e.finalNorm), whole(e.normed))
	e.buildLMShards()
	e.mkBG("argmax", p["argmax"], whole(e.uArgMax), whole(e.logits), whole(e.outTok))
	e.mkBG("inc_pos", p["inc_pos"], whole(e.step))
	e.mkBG("advance", p["advance"], whole(e.step), whole(e.outTok), whole(e.histBuf), whole(e.token))

	for i := range e.blocks {
		b := &e.blocks[i]
		tag := fmt.Sprintf("L%d", i)
		uDown := e.gemvU(tag+"_uDown", b.down.cols, b.down.rows)
		// FFN down projects straight into residual (skips separate resid pass).
		e.mkBG(tag+"_down", p["bingemv_add"], whole(uDown), whole(e.inter), whole(b.down.scales), whole(b.down.weights), whole(e.hidden))

		switch b.layerType {
		case "full_attention":
			// Wave-1: FFN pre-norm folded into SwiGLU (no rms2).
			uSW := e.uni(tag+"_uSWr", packU32(uint32(b.gate.cols), uint32(b.gate.rows), mathFloat32bits(e.eps), 0))
			e.mkBG(tag+"_sw", p["binswiglu_rms"], whole(uSW), whole(e.hidden), whole(b.ffnNorm),
				whole(b.gate.scales), whole(b.gate.weights),
				whole(b.up.scales), whole(b.up.weights), whole(e.inter))
			e.buildAttnBGs(tag, b)
		case "linear_attention":
			// GDN unchanged: separate rms1/rms2 + binswiglu.
			e.mkBG(tag+"_rms1", p["rmsnorm"], whole(e.uRMS), whole(e.hidden), whole(b.attnNorm), whole(e.normed))
			e.mkBG(tag+"_rms2", p["rmsnorm"], whole(e.uRMS), whole(e.hidden), whole(b.ffnNorm), whole(e.normed))
			uSW := e.uni(tag+"_uSW", packU32(uint32(b.gate.cols), uint32(b.gate.rows), 0, 0))
			e.mkBG(tag+"_sw", p["binswiglu"], whole(uSW), whole(e.normed),
				whole(b.gate.scales), whole(b.gate.weights),
				whole(b.up.scales), whole(b.up.weights), whole(e.inter))
			e.buildGDNBGs(tag, b)
		}
	}
}

func (e *hybridEngine) buildLMShards() {
	lim := MaxStorageBindingLimit()
	if lim == 0 {
		e.mkBG("lm", e.pipe["bingemv"], whole(e.uGemvVocabH), whole(e.normed), whole(e.lmHead.scales), whole(e.lmHead.weights), whole(e.logits))
		return
	}
	scaleRowBytes := uint64((e.lmHead.cols / 128) * 4)
	weightRowBytes := uint64((e.lmHead.cols / 32) * 4)
	logitRowBytes := uint64(4)
	if scaleRowBytes == 0 || weightRowBytes == 0 {
		e.mkBG("lm", e.pipe["bingemv"], whole(e.uGemvVocabH), whole(e.normed), whole(e.lmHead.scales), whole(e.lmHead.weights), whole(e.logits))
		return
	}
	maxRows := int(minU64(minU64(lim/scaleRowBytes, lim/weightRowBytes), lim/logitRowBytes))
	if maxRows <= 0 || maxRows >= e.vocabN {
		e.mkBG("lm", e.pipe["bingemv"], whole(e.uGemvVocabH), whole(e.normed), whole(e.lmHead.scales), whole(e.lmHead.weights), whole(e.logits))
		return
	}
	e.lmShards = make([]lmShard, 0, (e.vocabN+maxRows-1)/maxRows)
	for start := 0; start < e.vocabN; start += maxRows {
		rows := maxRows
		if rem := e.vocabN - start; rem < rows {
			rows = rem
		}
		u := e.uni(fmt.Sprintf("uLM_%d", start), packU32(uint32(e.hiddenN), uint32(rows), 0, 0))
		bg := e.mkBG(
			fmt.Sprintf("lm_%d", start),
			e.pipe["bingemv"],
			whole(u),
			whole(e.normed),
			bufSlice{buf: e.lmHead.scales, offset: uint64(start) * scaleRowBytes, size: uint64(rows) * scaleRowBytes},
			bufSlice{buf: e.lmHead.weights, offset: uint64(start) * weightRowBytes, size: uint64(rows) * weightRowBytes},
			bufSlice{buf: e.logits, offset: uint64(start) * logitRowBytes, size: uint64(rows) * logitRowBytes},
		)
		e.lmShards = append(e.lmShards, lmShard{bg: bg, workgroups: binWG(rows)})
	}
	fmt.Printf("  lm head sharded for maxSSBO=%d (%d shards)\n", lim, len(e.lmShards))
}

func (e *hybridEngine) dispatchLM(pass *wgpu.ComputePassEncoder) {
	if len(e.lmShards) == 0 {
		e.disp(pass, e.pipe["bingemv"], e.bg["lm"], binWG(e.vocabN), 1, 1)
		return
	}
	for _, s := range e.lmShards {
		e.disp(pass, e.pipe["bingemv"], s.bg, s.workgroups, 1, 1)
	}
}

func (e *hybridEngine) buildAttnBGs(tag string, b *hybridBlockGPU) {
	p := e.pipe
	if b.k.rows != b.v.rows || b.k.cols != b.v.cols {
		panic(fmt.Sprintf("%s: K/V shape mismatch for dual GEMV", tag))
	}
	if b.q.cols != b.k.cols {
		panic(fmt.Sprintf("%s: Q/K input cols mismatch for qkv_rms", tag))
	}
	uO := e.gemvU(tag+"_uO", b.o.cols, b.o.rows)

	qOut := e.qBuf
	if b.outputGate {
		qOut = e.qGate
	}
	uQKV := e.uni(tag+"_uQKV", packU32(uint32(b.q.cols), uint32(b.q.rows), uint32(b.k.rows), mathFloat32bits(e.eps)))
	e.mkBG(tag+"_qkv", p["bingemv_qkv_rms"], whole(uQKV), whole(e.hidden), whole(b.attnNorm),
		whole(b.q.scales), whole(b.q.weights),
		whole(b.k.scales), whole(b.k.weights),
		whole(b.v.scales), whole(b.v.weights),
		whole(qOut), whole(e.kBuf), whole(e.vBuf))

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
	og := uint32(0)
	if b.outputGate {
		og = 1
	}
	// qGate must be a distinct buffer from q: WebGPU forbids the same buffer as
	// STORAGE_READ_ONLY and STORAGE_READ_WRITE in one dispatch. When outputGate=0
	// the shader ignores qGate; still bind the dedicated qGate scratch.
	if e.qGate == nil {
		panic(fmt.Sprintf("%s: qGate scratch missing for attn_prep", tag))
	}
	uPrep := e.uni(tag+"_uPrep", packU32(
		uint32(b.numHeads), uint32(b.numKVHeads), uint32(b.headDim), mathFloat32bits(e.eps),
		uint32(rotDim), mathFloat32bits(theta), og, uint32(e.maxSeq),
	))
	e.mkBG(tag+"_prep", p["attn_prep"], whole(uPrep), whole(e.step), whole(e.qGate), whole(e.qBuf), whole(e.gateBuf),
		whole(b.qNorm), whole(e.kBuf), whole(e.vBuf), whole(b.kNorm), whole(b.kCache), whole(b.vCache))

	uAttn := e.uni(tag+"_uAttn", packU32(
		uint32(b.numHeads), uint32(b.numKVHeads), uint32(b.headDim), uint32(e.maxSeq),
		og, 0, 0, 0,
	))
	e.mkBG(tag+"_attn", p["attn_gated"], whole(uAttn), whole(e.step), whole(e.qBuf), whole(b.kCache), whole(b.vCache), whole(e.attnOut), whole(e.gateBuf))

	e.mkBG(tag+"_o", p["bingemv_add"], whole(uO), whole(e.attnOut), whole(b.o.scales), whole(b.o.weights), whole(e.hidden))
}

func (e *hybridEngine) buildGDNBGs(tag string, b *hybridBlockGPU) {
	p := e.pipe
	keyDim := b.numKeyHeads * b.keyHeadDim
	valDim := b.numValueHeads * b.valueHeadDim
	convDim := keyDim*2 + valDim

	uQKV := e.gemvU(tag+"_uQKV", b.gdnQKV.cols, b.gdnQKV.rows)
	uZ := e.gemvU(tag+"_uZ", b.gdnZ.cols, b.gdnZ.rows)
	if b.gdnB.rows != b.gdnA.rows || b.gdnB.cols != b.gdnA.cols {
		panic(fmt.Sprintf("%s: GDN B/A shape mismatch for dual GEMV", tag))
	}
	uBA := e.gemvU(tag+"_uBA", b.gdnB.cols, b.gdnB.rows)
	uOut := e.gemvU(tag+"_uOut", b.gdnOut.cols, b.gdnOut.rows)

	e.mkBG(tag+"_gqkv", p["bingemv"], whole(uQKV), whole(e.normed), whole(b.gdnQKV.scales), whole(b.gdnQKV.weights), whole(e.gdnQKV))
	e.mkBG(tag+"_gz", p["bingemv"], whole(uZ), whole(e.normed), whole(b.gdnZ.scales), whole(b.gdnZ.weights), whole(e.gdnZ))
	// B∥A are tiny (numV rows) but still reload full hidden — fuse like SwiGLU.
	e.mkBG(tag+"_gba", p["bingemv_dual"], whole(uBA), whole(e.normed),
		whole(b.gdnB.scales), whole(b.gdnB.weights),
		whole(b.gdnA.scales), whole(b.gdnA.weights),
		whole(e.gdnBetaRaw), whole(e.gdnARaw))

	uConv := e.uni(tag+"_uConv", packU32(uint32(convDim), uint32(b.convKernel), 0, 0))
	e.mkBG(tag+"_gconv", p["gdn_conv"], whole(uConv), whole(e.gdnQKV), whole(b.gdnConv), whole(b.gdnConvState), whole(e.gdnMixed))

	uPrep := e.uni(tag+"_uPrep", packU32(uint32(b.numKeyHeads), uint32(b.numValueHeads), uint32(b.keyHeadDim), uint32(b.valueHeadDim)))
	e.mkBG(tag+"_gprep", p["gdn_prep"], whole(uPrep), whole(e.gdnMixed), whole(e.gdnQRep), whole(e.gdnKRep),
		whole(e.gdnBetaRaw), whole(e.gdnARaw), whole(b.gdnALog), whole(b.gdnDtBias), whole(e.gdnBeta), whole(e.gdnG))

	// Fused recurrent step + gate/RMSNorm (skips a second dispatch).
	uStep := e.uni(tag+"_uStep", packU32(uint32(b.numValueHeads), uint32(b.keyHeadDim), uint32(b.valueHeadDim), mathFloat32bits(e.eps)))
	vOff := uint64(keyDim * 2 * 4)
	vSize := uint64(valDim * 4)
	e.mkBG(tag+"_gstep", p["gdn_step"], whole(uStep), whole(e.gdnQRep), whole(e.gdnKRep),
		bufSlice{e.gdnMixed, vOff, vSize}, whole(e.gdnBeta), whole(e.gdnG), whole(b.gdnState), whole(e.gdnCore),
		whole(e.gdnZ), whole(b.gdnNorm))
	// GDN out-proj accumulates into residual.
	e.mkBG(tag+"_gout", p["bingemv_add"], whole(uOut), whole(e.gdnCore), whole(b.gdnOut.scales), whole(b.gdnOut.weights), whole(e.hidden))
}

func (e *hybridEngine) disp(pass *wgpu.ComputePassEncoder, pipe *wgpu.ComputePipeline, bg *wgpu.BindGroup, x, y, z uint32) {
	e.dispCount++
	pass.SetPipeline(pipe)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(x, y, z)
}

func (e *hybridEngine) beginTokenDisp() { e.dispCount = 0 }
func (e *hybridEngine) endTokenDisp()   { noteHybridStats(e.dispCount, e.useSPIRV) }

func binWG(rows int) uint32 { return (uint32(rows) + 127) / 128 }

func (e *hybridEngine) recordLayers(pass *wgpu.ComputePassEncoder) {
	p := e.pipe
	iWG := binWG(e.interN)
	for i := range e.blocks {
		b := &e.blocks[i]
		tag := fmt.Sprintf("L%d", i)

		switch b.layerType {
		case "full_attention":
			qkvWG := binWG(b.q.rows)
			if kv := binWG(b.k.rows); kv > qkvWG {
				qkvWG = kv
			}
			e.disp(pass, p["bingemv_qkv_rms"], e.bg[tag+"_qkv"], qkvWG, 1, 1)
			e.disp(pass, p["attn_prep"], e.bg[tag+"_prep"], uint32(b.numHeads), 1, 1)
			e.disp(pass, p["attn_gated"], e.bg[tag+"_attn"], uint32(b.numHeads), 1, 1)
			e.disp(pass, p["bingemv_add"], e.bg[tag+"_o"], binWG(b.o.rows), 1, 1)
			e.disp(pass, p["binswiglu_rms"], e.bg[tag+"_sw"], iWG, 1, 1)
			e.disp(pass, p["bingemv_add"], e.bg[tag+"_down"], binWG(b.down.rows), 1, 1)
		case "linear_attention":
			e.disp(pass, p["rmsnorm"], e.bg[tag+"_rms1"], 1, 1, 1)
			keyDim := b.numKeyHeads * b.keyHeadDim
			convDim := keyDim*2 + b.numValueHeads*b.valueHeadDim
			prepWG := uint32(b.numValueHeads)
			if uint32(b.numKeyHeads) > prepWG {
				prepWG = uint32(b.numKeyHeads)
			}
			e.disp(pass, p["bingemv"], e.bg[tag+"_gqkv"], binWG(b.gdnQKV.rows), 1, 1)
			e.disp(pass, p["bingemv"], e.bg[tag+"_gz"], binWG(b.gdnZ.rows), 1, 1)
			e.disp(pass, p["bingemv_dual"], e.bg[tag+"_gba"], binWG(b.gdnB.rows), 1, 1)
			e.disp(pass, p["gdn_conv"], e.bg[tag+"_gconv"], (uint32(convDim)+63)/64, 1, 1)
			e.disp(pass, p["gdn_prep"], e.bg[tag+"_gprep"], prepWG, 1, 1)
			e.disp(pass, p["gdn_step"], e.bg[tag+"_gstep"], uint32(b.numValueHeads), 1, 1)
			e.disp(pass, p["bingemv_add"], e.bg[tag+"_gout"], binWG(b.gdnOut.rows), 1, 1)
			e.disp(pass, p["rmsnorm"], e.bg[tag+"_rms2"], 1, 1, 1)
			e.disp(pass, p["binswiglu"], e.bg[tag+"_sw"], iWG, 1, 1)
			e.disp(pass, p["bingemv_add"], e.bg[tag+"_down"], binWG(b.down.rows), 1, 1)
		}
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
	e.beginTokenDisp()
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
		e.dispatchLM(pass)
	}
	e.disp(pass, e.pipe["inc_pos"], e.bg["inc_pos"], 1, 1, 1)
	pass.End()
	e.endTokenDisp()

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

func (e *hybridEngine) recordSample(pass *wgpu.ComputePassEncoder) {
	p := e.pipe
	e.dispatchLM(pass)
	e.disp(pass, p["argmax"], e.bg["argmax"], 1, 1, 1)
	e.disp(pass, p["advance"], e.bg["advance"], 1, 1, 1)
}

// stepTokenSample runs one forward + LM + argmax + advance; maps 4 bytes from hist.
func (e *hybridEngine) stepTokenSample(id uint32) (uint32, error) {
	e.beginTokenDisp()
	e.queue.WriteBuffer(e.token, 0, packU32(id))
	e.queue.WriteBuffer(e.step, 0, packU32(uint32(e.pos), 0))

	enc, err := e.device.CreateCommandEncoder(nil)
	if err != nil {
		return 0, err
	}
	pass := enc.BeginComputePass(nil)
	e.disp(pass, e.pipe["binembed"], e.bg["embed"], (uint32(e.hiddenN)+63)/64, 1, 1)
	e.recordLayers(pass)
	e.recordSample(pass)
	pass.End()
	e.endTokenDisp()
	toks, err := e.runHist(enc, 1)
	if err != nil {
		return 0, err
	}
	e.pos++
	return toks[0], nil
}

// decodeChunkSample runs k decode steps in one submit (embed←token, sample+advance each).
// GPU token/pos must already be valid (typically after PrefillSample / prior chunk).
func (e *hybridEngine) decodeChunkSample(k int) ([]uint32, error) {
	if k < 1 {
		return nil, fmt.Errorf("fusedgpu: decode chunk k < 1")
	}
	if e.pos+k > e.maxSeq {
		return nil, fmt.Errorf("fusedgpu: decode would exceed maxSeq %d (pos=%d k=%d)", e.maxSeq, e.pos, k)
	}
	// Keep GPU pos; reset hist write index so this chunk packs contiguously.
	e.queue.WriteBuffer(e.step, 0, packU32(uint32(e.pos), 0))

	if runtime.GOOS == "android" {
		// gpu_multi_fuse: one full forward per command buffer. Packing k>1 into a
		// single pass trips Turnip device-lost → wgpuQueueSubmitForIndex abort.
		out := make([]uint32, 0, k)
		for i := 0; i < k; i++ {
			e.beginTokenDisp()
			enc, err := e.device.CreateCommandEncoder(nil)
			if err != nil {
				return nil, err
			}
			pass := enc.BeginComputePass(nil)
			e.disp(pass, e.pipe["binembed"], e.bg["embed"], (uint32(e.hiddenN)+63)/64, 1, 1)
			e.recordLayers(pass)
			e.recordSample(pass)
			pass.End()
			e.endTokenDisp()
			toks, err := e.runHist(enc, 1)
			if err != nil {
				return nil, fmt.Errorf("decode step %d/%d: %w", i+1, k, err)
			}
			out = append(out, toks[0])
			e.pos++
		}
		return out, nil
	}

	enc, err := e.device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, err
	}
	e.beginTokenDisp()
	pass := enc.BeginComputePass(nil)
	for i := 0; i < k; i++ {
		e.disp(pass, e.pipe["binembed"], e.bg["embed"], (uint32(e.hiddenN)+63)/64, 1, 1)
		e.recordLayers(pass)
		e.recordSample(pass)
	}
	pass.End()
	if k > 0 {
		noteHybridStats(e.dispCount/k, e.useSPIRV)
	}
	toks, err := e.runHist(enc, k)
	if err != nil {
		return nil, err
	}
	e.pos += k
	return toks, nil
}

// prefillSample runs prompt tokens in one GPU submit; argmax+advance on last.
func (e *hybridEngine) prefillSample(ids []uint32) (uint32, error) {
	if len(ids) == 0 {
		return 0, fmt.Errorf("fusedgpu: empty ids")
	}
	if len(ids) > e.maxSeq {
		return 0, fmt.Errorf("fusedgpu: prompt len %d > maxSeq %d", len(ids), e.maxSeq)
	}
	if len(ids) == 1 {
		// Single-token prompt: embed from token path + sample.
		e.pos = 0
		return e.stepTokenSample(ids[0])
	}
	if runtime.GOOS == "android" {
		// gpu_multi_fuse command sharding: one token → one command buffer.
		// Multi-token packing (desktop gpu_fuse) device-lost aborts on Turnip A710.
		e.pos = 0
		for i := 0; i < len(ids)-1; i++ {
			if err := e.stepToken(ids[i], false, nil); err != nil {
				return 0, fmt.Errorf("prefill step %d: %w", i, err)
			}
		}
		return e.stepTokenSample(ids[len(ids)-1])
	}

	e.queue.WriteBuffer(e.promptBuf, 0, u32Bytes(ids))
	e.queue.WriteBuffer(e.step, 0, packU32(0, 0))
	e.pos = 0
	n := len(ids)
	prefillChunk := 128
	for base := 0; base < n; base += prefillChunk {
		end := base + prefillChunk
		if end > n {
			end = n
		}
		enc, err := e.device.CreateCommandEncoder(nil)
		if err != nil {
			return 0, err
		}
		pass := enc.BeginComputePass(nil)
		for i := base; i < end; i++ {
			e.disp(pass, e.pipe["binembed_p"], e.bg["embed_p"], (uint32(e.hiddenN)+63)/64, 1, 1)
			e.recordLayers(pass)
			if i+1 < n {
				e.disp(pass, e.pipe["inc_pos"], e.bg["inc_pos"], 1, 1, 1)
			} else {
				e.recordSample(pass)
			}
		}
		pass.End()
		if end < n {
			cmd, err := enc.Finish(nil)
			if err != nil {
				return 0, err
			}
			e.queue.Submit(cmd)
			continue
		}
		toks, err := e.runHist(enc, 1)
		if err != nil {
			return 0, err
		}
		e.pos = n
		return toks[0], nil
	}
	return 0, fmt.Errorf("fusedgpu: prefill produced no token")
}

func (e *hybridEngine) runHist(enc *wgpu.CommandEncoder, histCount int) ([]uint32, error) {
	bytes := uint64(histCount * 4)
	if bytes < 4 {
		bytes = 4
	}
	enc.CopyBufferToBuffer(e.histBuf, 0, e.stagingHist, 0, bytes)
	cmd, err := enc.Finish(nil)
	if err != nil {
		return nil, err
	}
	e.queue.Submit(cmd)

	done := make(chan struct{})
	var st wgpu.BufferMapAsyncStatus
	if err := e.stagingHist.MapAsync(wgpu.MapModeRead, 0, bytes, func(status wgpu.BufferMapAsyncStatus) {
		st = status
		close(done)
	}); err != nil {
		return nil, err
	}
	deadline := time.Now().Add(120 * time.Second)
	if runtime.GOOS == "android" {
		deadline = time.Now().Add(45 * time.Second)
	}
	for {
		e.device.Poll(false, nil)
		select {
		case <-done:
			if st != wgpu.BufferMapAsyncStatusSuccess {
				return nil, fmt.Errorf("fusedgpu hybrid hist MapAsync %v", st)
			}
			raw := e.stagingHist.GetMappedRange(0, uint(bytes))
			out := make([]uint32, histCount)
			for i := 0; i < histCount; i++ {
				out[i] = binary.LittleEndian.Uint32(raw[i*4 : i*4+4])
			}
			e.stagingHist.Unmap()
			return out, nil
		default:
			if time.Now().After(deadline) {
				return nil, fmt.Errorf("fusedgpu hybrid hist MapAsync timeout")
			}
			runtime.Gosched()
		}
	}
}

func (e *hybridEngine) readTok() (uint32, error) {
	const bytes = 4
	done := make(chan struct{})
	var st wgpu.BufferMapAsyncStatus
	if err := e.stagingTok.MapAsync(wgpu.MapModeRead, 0, bytes, func(status wgpu.BufferMapAsyncStatus) {
		st = status
		close(done)
	}); err != nil {
		return 0, err
	}
	deadline := time.Now().Add(120 * time.Second)
	for {
		e.device.Poll(false, nil)
		select {
		case <-done:
			if st != wgpu.BufferMapAsyncStatusSuccess {
				return 0, fmt.Errorf("fusedgpu hybrid tok MapAsync %v", st)
			}
			raw := e.stagingTok.GetMappedRange(0, bytes)
			tok := binary.LittleEndian.Uint32(raw)
			e.stagingTok.Unmap()
			return tok, nil
		default:
			if time.Now().After(deadline) {
				return 0, fmt.Errorf("fusedgpu hybrid tok MapAsync timeout")
			}
			runtime.Gosched()
		}
	}
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
