//go:build birdkit_native_vk && android && cgo

package fusedgpu

/*
#include <vulkan/vulkan.h>
*/
import "C"

import (
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"runtime/debug"
	"sort"
	"unsafe"
)

// Pipelines that must have AOT SPIR-V before native Vulkan hybrid can run.
var vkRequiredSPIRV = []string{
	"bingemv", "bingemv_add", "bingemv_dual", "bingemv_qkv_rms",
	"binswiglu", "binswiglu_rms", "binembed", "rmsnorm",
	"attn_prep", "attn_gated", "argmax", "advance", "inc_pos", "zero",
}

// vkHybridPipeLayout: per-binding descriptor kinds matching AOT SPIR-V
// (U = uniform buffer, S = storage buffer). Derived from naga output —
// note inc_pos/advance have no UBO (binding 0 is storage).
var vkHybridPipeLayout = map[string]string{
	"bingemv":         "USSSS",
	"bingemv_add":     "USSSS",
	"bingemv_dual":    "USSSSSSS", // 8
	"bingemv_qkv_rms": "USSSSSSSSSSS", // 12
	"binswiglu":       "USSSSSS",      // 7
	"binswiglu_rms":   "USSSSSSS",     // 8
	"binembed":        "USSSS",
	"binembed_p":      "USSSSS",
	"rmsnorm":         "USSS",
	"attn_q_prep":     "USSSSS",
	"attn_kv_prep":    "USSSSSS",
	"attn_prep":       "USSSSSSSSSS", // 11
	"attn_gated":      "USSSSSS",
	"inc_pos":         "S",
	"advance":         "SSSS",
	"argmax":          "USS",
	"zero":            "US",
}

func vkDescTypesFromLayout(layout string) ([]C.uint32_t, error) {
	if layout == "" {
		return nil, fmt.Errorf("empty layout")
	}
	out := make([]C.uint32_t, len(layout))
	for i := 0; i < len(layout); i++ {
		switch layout[i] {
		case 'U':
			out[i] = C.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
		case 'S':
			out[i] = C.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
		default:
			return nil, fmt.Errorf("bad layout char %q", layout[i])
		}
	}
	return out, nil
}

type vkBinGPU struct {
	scales, weights *vkBuffer
	rows, cols      int
}

type vkHybridBlockGPU struct {
	layerType string
	attnNorm  *vkBuffer
	ffnNorm   *vkBuffer
	gate, up, down vkBinGPU
	q, k, v, o     vkBinGPU
	qNorm, kNorm   *vkBuffer
	kCache, vCache *vkBuffer
	outputGate     bool
	partialRotary  float32
	ropeTheta      float32
	numHeads       int
	numKVHeads     int
	headDim        int
	qRows          int
}

type vkDescSet struct {
	set uintptr
}

type hybridVKEngine struct {
	dev *vkDevice

	pipe map[string]*vkPipelineBundle
	bg   map[string]vkDescSet
	owned []*vkBuffer

	spec *HybridSpec

	embed, lmHead vkBinGPU
	finalNorm     *vkBuffer
	blocks        []vkHybridBlockGPU

	step, token         *vkBuffer
	promptBuf           *vkBuffer
	histBuf             *vkBuffer
	stagingHist         *vkBuffer
	hidden, normed, mix *vkBuffer
	inter, upBuf        *vkBuffer
	logits              *vkBuffer
	outTok              *vkBuffer

	qGate, qBuf, gateBuf, kBuf, vBuf, attnOut *vkBuffer

	uRMS, uResidH, uSwiglu   *vkBuffer
	uEmbed                   *vkBuffer
	uGemvVocabH              *vkBuffer
	uGemvHInter, uGemvInterH *vkBuffer
	uArgMax                  *vkBuffer

	hiddenN, vocabN, interN, maxSeq int
	eps                             float32
	pos                             int
	adapterName                     string

	maxQDim, maxKVDim, maxQGate int
	dispCount                   int
}

func NewHybridVKFromSpec(spec *HybridSpec) (*HybridVKEngine, error) {
	if spec == nil {
		return nil, fmt.Errorf("fusedgpu: nil hybrid spec")
	}
	if spec.Layers <= 0 || len(spec.Blocks) != spec.Layers {
		return nil, fmt.Errorf("fusedgpu: hybrid block count mismatch")
	}
	if spec.Intermediate <= 0 {
		return nil, fmt.Errorf("fusedgpu: hybrid Intermediate unset")
	}
	for i := range spec.Blocks {
		if spec.Blocks[i].LayerType == "linear_attention" {
			return nil, fmt.Errorf("fusedgpu: GDN not yet on VK (layer %d)", i)
		}
	}
	e, err := newHybridVKEngine(spec)
	if err != nil {
		return nil, err
	}
	spec.clearPayloads()
	fmt.Printf("✅ Hybrid native Vulkan ready (~%.1f GiB) on %s\n", float64(e.estimateVRAM())/(1<<30), e.adapterName)
	return &HybridVKEngine{e: e}, nil
}

func newHybridVKEngine(spec *HybridSpec) (*hybridVKEngine, error) {
	dev, err := initVKDevice()
	if err != nil {
		return nil, err
	}
	e := &hybridVKEngine{
		dev: dev, pipe: map[string]*vkPipelineBundle{}, bg: map[string]vkDescSet{},
		spec: spec, hiddenN: spec.Hidden, vocabN: spec.Vocab, interN: spec.Intermediate,
		maxSeq: spec.MaxSeq, eps: spec.Eps, adapterName: dev.name,
	}
	if e.maxSeq <= 0 {
		e.maxSeq = DefaultMaxSeq
	}
	e.maxSeq = ClampAttnMaxSeq(e.maxSeq)
	if e.eps <= 0 {
		e.eps = 1e-6
	}
	e.deriveMaxDims(spec)
	for _, name := range vkRequiredSPIRV {
		spv, ok := hybridSPIRV[name]
		if !ok || len(spv) < 4 {
			dev.destroy()
			return nil, fmt.Errorf("pipeline %s: missing SPIR-V (run scripts/compile-hybrid-spirv.sh)", name)
		}
	}
	// descriptor pool: generous for all layer bind groups
	if err := dev.allocDescPool(4096, 512, 4096); err != nil {
		dev.destroy()
		return nil, err
	}
	spirvHits := 0
	// Deterministic order for logcat debugging.
	names := make([]string, 0, len(vkHybridPipeLayout))
	for name := range vkHybridPipeLayout {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		layout := vkHybridPipeLayout[name]
		spv, ok := hybridSPIRV[name]
		if !ok || len(spv) < 4 {
			continue
		}
		types, err := vkDescTypesFromLayout(layout)
		if err != nil {
			e.release()
			return nil, fmt.Errorf("pipeline %s layout: %w", name, err)
		}
		fmt.Printf("  hybrid vk create pipeline %s layout=%s spirv=%d\n", name, layout, len(spv))
		vkALog(fmt.Sprintf("create pipeline %s layout=%s spirv=%d", name, layout, len(spv)))
		p, err := dev.createComputePipeline(name, spv, types)
		if err != nil {
			vkALog(fmt.Sprintf("pipeline %s FAILED: %v", name, err))
			e.release()
			return nil, fmt.Errorf("pipeline %s: %w", name, err)
		}
		vkALog(fmt.Sprintf("pipeline %s OK", name))
		e.pipe[name] = p
		spirvHits++
	}
	if spirvHits == 0 {
		e.release()
		return nil, fmt.Errorf("no SPIR-V pipelines loaded")
	}
	vkALog(fmt.Sprintf("pipelines ready count=%d; uploading weights…", spirvHits))
	if err := e.uploadAll(spec); err != nil {
		vkALog(fmt.Sprintf("upload FAILED: %v", err))
		e.release()
		return nil, err
	}
	vkALog("upload OK; alloc scratch…")
	if err := e.allocScratch(); err != nil {
		vkALog(fmt.Sprintf("scratch FAILED: %v", err))
		e.release()
		return nil, err
	}
	e.initUniforms()
	if err := e.buildBindGroups(); err != nil {
		vkALog(fmt.Sprintf("bindgroups FAILED: %v", err))
		e.release()
		return nil, err
	}
	vkALog(fmt.Sprintf("Hybrid VK ready spirv=%d on %s", spirvHits, e.adapterName))
	fmt.Printf("  Hybrid VK spirv=%d pipelines\n", spirvHits)
	return e, nil
}

func (e *hybridVKEngine) deriveMaxDims(spec *HybridSpec) {
	for i := range spec.Blocks {
		b := &spec.Blocks[i]
		if b.LayerType != "full_attention" {
			continue
		}
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
}

func (e *hybridVKEngine) estimateVRAM() uint64 {
	var n uint64
	for _, b := range e.owned {
		if b != nil {
			n += b.size
		}
	}
	return n
}

func (e *hybridVKEngine) mkBuf(label string, size uint64, storage bool, data []byte) (*vkBuffer, error) {
	size = vkAlignSize(size)
	if storage && e.dev.maxSSBO > 0 && size > e.dev.maxSSBO {
		return nil, fmt.Errorf("storage buffer %s %d > maxSSBO", label, size)
	}
	b, err := e.dev.createBuffer(label, size, storage, true, data)
	if err != nil {
		return nil, err
	}
	e.owned = append(e.owned, b)
	return b, nil
}

func (e *hybridVKEngine) uploadBin(label string, s *BinarySpec) (vkBinGPU, error) {
	if s == nil || s.Rows <= 0 || s.Cols <= 0 || len(s.Words) == 0 {
		return vkBinGPU{}, fmt.Errorf("%s: empty binary matrix", label)
	}
	sc, err := e.mkBuf(label+"_s", uint64(len(s.Scales)*4), true, f32Bytes(s.Scales))
	if err != nil {
		return vkBinGPU{}, err
	}
	w, err := e.mkBuf(label+"_w", uint64(len(s.Words)*4), true, u32Bytes(s.Words))
	if err != nil {
		return vkBinGPU{}, err
	}
	s.Scales, s.Words = nil, nil
	return vkBinGPU{scales: sc, weights: w, rows: s.Rows, cols: s.Cols}, nil
}

func (e *hybridVKEngine) uploadAll(spec *HybridSpec) error {
	var err error
	if e.embed, err = e.uploadBin("embed", &spec.Embed); err != nil {
		return err
	}
	if spec.LMHeadTied {
		e.lmHead = e.embed
	} else if e.lmHead, err = e.uploadBin("lm", &spec.LMHead); err != nil {
		return err
	}
	fn := spec.FinalNorm
	if len(fn) == 0 {
		fn = onesF32Hybrid(spec.Hidden)
	}
	if e.finalNorm, err = e.mkBuf("fnorm", uint64(len(fn)*4), true, f32Bytes(fn)); err != nil {
		return err
	}
	spec.FinalNorm = nil

	e.blocks = make([]vkHybridBlockGPU, spec.Layers)
	for i := range spec.Blocks {
		b := &spec.Blocks[i]
		g := &e.blocks[i]
		g.layerType = b.LayerType
		if g.attnNorm, err = e.mkBuf(fmt.Sprintf("an_%d", i), uint64(len(b.AttnNorm)*4), true, f32Bytes(b.AttnNorm)); err != nil {
			return err
		}
		if g.ffnNorm, err = e.mkBuf(fmt.Sprintf("fn_%d", i), uint64(len(b.FFNNorm)*4), true, f32Bytes(b.FFNNorm)); err != nil {
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
		if b.LayerType != "full_attention" {
			return fmt.Errorf("hybrid layer %d: unsupported type %q on VK", i, b.LayerType)
		}
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
		if g.qNorm, err = e.mkBuf(fmt.Sprintf("qn_%d", i), uint64(len(b.QNorm)*4), true, f32Bytes(b.QNorm)); err != nil {
			return err
		}
		if g.kNorm, err = e.mkBuf(fmt.Sprintf("kn_%d", i), uint64(len(b.KNorm)*4), true, f32Bytes(b.KNorm)); err != nil {
			return err
		}
		b.QNorm, b.KNorm = nil, nil
		kvBytes := uint64(b.NumKVHeads * e.maxSeq * b.HeadDim * 4)
		if g.kCache, err = e.mkBuf(fmt.Sprintf("kc_%d", i), kvBytes, true, nil); err != nil {
			return err
		}
		if g.vCache, err = e.mkBuf(fmt.Sprintf("vc_%d", i), kvBytes, true, nil); err != nil {
			return err
		}
		if (i+1)%8 == 0 || i+1 == spec.Layers {
			fmt.Printf("  hybrid vk upload layers %d/%d\n", i+1, spec.Layers)
			runtime.GC()
			debug.FreeOSMemory()
		}
	}
	return nil
}

func (e *hybridVKEngine) allocScratch() error {
	H := uint64(e.hiddenN * 4)
	var err error
	if e.step, err = e.mkBuf("step", 64, true, nil); err != nil {
		return err
	}
	if e.token, err = e.mkBuf("token", 64, true, nil); err != nil {
		return err
	}
	if e.promptBuf, err = e.mkBuf("prompt", uint64(e.maxSeq*4), true, nil); err != nil {
		return err
	}
	if e.hidden, err = e.mkBuf("h", H, true, nil); err != nil {
		return err
	}
	if e.normed, err = e.mkBuf("norm", H, true, nil); err != nil {
		return err
	}
	if e.mix, err = e.mkBuf("mix", H, true, nil); err != nil {
		return err
	}
	if e.inter, err = e.mkBuf("inter", uint64(e.interN*4), true, nil); err != nil {
		return err
	}
	if e.upBuf, err = e.mkBuf("upbuf", uint64(e.interN*4), true, nil); err != nil {
		return err
	}
	if e.logits, err = e.mkBuf("logits", uint64(e.vocabN*4), true, nil); err != nil {
		return err
	}
	if e.outTok, err = e.mkBuf("outTok", 64, true, nil); err != nil {
		return err
	}
	if e.histBuf, err = e.mkBuf("hist", uint64(e.maxSeq*4), true, nil); err != nil {
		return err
	}
	if e.stagingHist, err = e.mkBuf("stageHist", uint64(e.maxSeq*4), true, nil); err != nil {
		return err
	}
	if e.maxQGate > 0 {
		if e.qGate, err = e.mkBuf("qgate", uint64(e.maxQGate*4), true, nil); err != nil {
			return err
		}
		if e.qBuf, err = e.mkBuf("q", uint64(e.maxQDim*4), true, nil); err != nil {
			return err
		}
		if e.gateBuf, err = e.mkBuf("gate", uint64(e.maxQDim*4), true, nil); err != nil {
			return err
		}
		if e.kBuf, err = e.mkBuf("k", uint64(e.maxKVDim*4), true, nil); err != nil {
			return err
		}
		if e.vBuf, err = e.mkBuf("v", uint64(e.maxKVDim*4), true, nil); err != nil {
			return err
		}
		if e.attnOut, err = e.mkBuf("ao", uint64(e.maxQDim*4), true, nil); err != nil {
			return err
		}
	}
	return nil
}

func (e *hybridVKEngine) uni(label string, bytes []byte) *vkBuffer {
	b, err := e.mkBuf(label, 256, false, bytes)
	if err != nil {
		panic(err)
	}
	return b
}

func (e *hybridVKEngine) initUniforms() {
	e.uRMS = e.uni("uRMS", packMix(uint32(e.hiddenN), e.eps, 0, 0))
	e.uResidH = e.uni("uRH", packU32(uint32(e.hiddenN), 0, 0, 0))
	e.uSwiglu = e.uni("uSW", packU32(uint32(e.hiddenN), uint32(e.interN), 0, 0))
	e.uEmbed = e.uni("uEM", packU32(uint32(e.hiddenN), uint32(e.embed.cols/32), uint32(e.embed.cols/128), 0))
	e.uGemvVocabH = e.uni("uVH", packU32(uint32(e.hiddenN), uint32(e.vocabN), 0, 0))
	e.uGemvHInter = e.uni("uDown", packU32(uint32(e.interN), uint32(e.hiddenN), 0, 0))
	e.uGemvInterH = e.uni("uGate", packU32(uint32(e.hiddenN), uint32(e.interN), 0, 0))
	e.uArgMax = e.uni("uAM", packU32(uint32(e.vocabN), 0, 0, 0))
}

func (e *hybridVKEngine) mkBG(key, pipeName string, bufs ...*vkBuffer) error {
	if _, ok := e.bg[key]; ok {
		return nil
	}
	p := e.pipe[pipeName]
	if p == nil {
		return fmt.Errorf("bindgroup %s: missing pipe %s", key, pipeName)
	}
	if len(bufs) != len(p.types) {
		return fmt.Errorf("bindgroup %s: got %d bufs want %d for %s", key, len(bufs), len(p.types), pipeName)
	}
	set, err := e.dev.allocDescSet(p.setLayout)
	if err != nil {
		return fmt.Errorf("bindgroup %s: %w", key, err)
	}
	e.dev.writeDescriptorSet(set, bufs, nil, p.types)
	e.bg[key] = vkDescSet{set: set}
	return nil
}

func (e *hybridVKEngine) gemvU(label string, cols, rows int) *vkBuffer {
	return e.uni(label, packU32(uint32(cols), uint32(rows), 0, 0))
}

func (e *hybridVKEngine) buildBindGroups() error {
	if err := e.mkBG("embed", "binembed", e.uEmbed, e.token, e.embed.scales, e.embed.weights, e.hidden); err != nil {
		return err
	}
	if err := e.mkBG("embed_p", "binembed_p", e.uEmbed, e.step, e.promptBuf, e.embed.scales, e.embed.weights, e.hidden); err != nil {
		return err
	}
	if err := e.mkBG("fnorm", "rmsnorm", e.uRMS, e.hidden, e.finalNorm, e.normed); err != nil {
		return err
	}
	if err := e.mkBG("lm", "bingemv", e.uGemvVocabH, e.normed, e.lmHead.scales, e.lmHead.weights, e.logits); err != nil {
		return err
	}
	if err := e.mkBG("argmax", "argmax", e.uArgMax, e.logits, e.outTok); err != nil {
		return err
	}
	if err := e.mkBG("inc_pos", "inc_pos", e.step); err != nil {
		return err
	}
	if err := e.mkBG("advance", "advance", e.step, e.outTok, e.histBuf, e.token); err != nil {
		return err
	}
	for i := range e.blocks {
		b := &e.blocks[i]
		tag := fmt.Sprintf("L%d", i)
		uDown := e.gemvU(tag+"_uDown", b.down.cols, b.down.rows)
		if err := e.mkBG(tag+"_down", "bingemv_add", uDown, e.inter, b.down.scales, b.down.weights, e.hidden); err != nil {
			return err
		}
		uSW := e.uni(tag+"_uSWr", packU32(uint32(b.gate.cols), uint32(b.gate.rows), mathFloat32bits(e.eps), 0))
		if err := e.mkBG(tag+"_sw", "binswiglu_rms", uSW, e.hidden, b.ffnNorm, b.gate.scales, b.gate.weights, b.up.scales, b.up.weights, e.inter); err != nil {
			return err
		}
		if err := e.buildAttnBGs(tag, b); err != nil {
			return err
		}
	}
	return nil
}

func (e *hybridVKEngine) buildAttnBGs(tag string, b *vkHybridBlockGPU) error {
	uO := e.gemvU(tag+"_uO", b.o.cols, b.o.rows)
	qOut := e.qBuf
	if b.outputGate {
		qOut = e.qGate
	}
	uQKV := e.uni(tag+"_uQKV", packU32(uint32(b.q.cols), uint32(b.q.rows), uint32(b.k.rows), mathFloat32bits(e.eps)))
	if err := e.mkBG(tag+"_qkv", "bingemv_qkv_rms", uQKV, e.hidden, b.attnNorm,
		b.q.scales, b.q.weights, b.k.scales, b.k.weights, b.v.scales, b.v.weights,
		qOut, e.kBuf, e.vBuf); err != nil {
		return err
	}
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
	qGateSrc := e.qGate
	if qGateSrc == nil {
		return fmt.Errorf("%s: qGate scratch missing for attn_prep", tag)
	}
	uPrep := e.uni(tag+"_uPrep", packU32(
		uint32(b.numHeads), uint32(b.numKVHeads), uint32(b.headDim), mathFloat32bits(e.eps),
		uint32(rotDim), mathFloat32bits(theta), og, uint32(e.maxSeq),
	))
	if err := e.mkBG(tag+"_prep", "attn_prep", uPrep, e.step, qGateSrc, e.qBuf, e.gateBuf,
		b.qNorm, e.kBuf, e.vBuf, b.kNorm, b.kCache, b.vCache); err != nil {
		return err
	}
	uAttn := e.uni(tag+"_uAttn", packU32(uint32(b.numHeads), uint32(b.numKVHeads), uint32(b.headDim), uint32(e.maxSeq), og, 0, 0, 0))
	if err := e.mkBG(tag+"_attn", "attn_gated", uAttn, e.step, e.qBuf, b.kCache, b.vCache, e.attnOut, e.gateBuf); err != nil {
		return err
	}
	return e.mkBG(tag+"_o", "bingemv_add", uO, e.attnOut, b.o.scales, b.o.weights, e.hidden)
}

func (e *hybridVKEngine) writeU32(buf *vkBuffer, off uint64, v uint32) {
	if buf.mapped == nil {
		return
	}
	*(*uint32)(unsafe.Pointer(uintptr(buf.mapped) + uintptr(off))) = v
}

func (e *hybridVKEngine) writeStep(pos, outCount uint32) {
	e.writeU32(e.step, 0, pos)
	e.writeU32(e.step, 4, outCount)
}

func (e *hybridVKEngine) bar(cmd uintptr) {
	e.dev.computeBarrier(cmd)
}

func (e *hybridVKEngine) disp(cmd uintptr, pipeName, bgKey string, x, y, z uint32) {
	e.dispCount++
	p := e.pipe[pipeName]
	bg := e.bg[bgKey]
	e.dev.bindDispatch(cmd, p.pipe, p.layout, bg.set, x, y, z)
}

// recordLayers emits one transformer stack with barriers only on true write→read edges.
func (e *hybridVKEngine) recordLayers(cmd uintptr) {
	iWG := binWG(e.interN)
	for i := range e.blocks {
		b := &e.blocks[i]
		tag := fmt.Sprintf("L%d", i)
		qkvWG := binWG(b.q.rows)
		if kv := binWG(b.k.rows); kv > qkvWG {
			qkvWG = kv
		}
		e.disp(cmd, "bingemv_qkv_rms", tag+"_qkv", qkvWG, 1, 1)
		e.bar(cmd)
		e.disp(cmd, "attn_prep", tag+"_prep", uint32(b.numHeads), 1, 1)
		e.bar(cmd)
		e.disp(cmd, "attn_gated", tag+"_attn", uint32(b.numHeads), 1, 1)
		e.bar(cmd)
		e.disp(cmd, "bingemv_add", tag+"_o", binWG(b.o.rows), 1, 1)
		e.bar(cmd)
		e.disp(cmd, "binswiglu_rms", tag+"_sw", iWG, 1, 1)
		e.bar(cmd)
		e.disp(cmd, "bingemv_add", tag+"_down", binWG(b.down.rows), 1, 1)
		e.bar(cmd)
	}
	e.disp(cmd, "rmsnorm", "fnorm", 1, 1, 1)
	e.bar(cmd)
}

func (e *hybridVKEngine) dispatchLM(cmd uintptr) {
	e.disp(cmd, "bingemv", "lm", binWG(e.vocabN), 1, 1)
	e.bar(cmd)
}

func (e *hybridVKEngine) recordSample(cmd uintptr) {
	e.dispatchLM(cmd)
	e.disp(cmd, "argmax", "argmax", 1, 1, 1)
	e.bar(cmd)
	e.disp(cmd, "advance", "advance", 1, 1, 1)
	e.bar(cmd)
}

func (e *hybridVKEngine) recordEmbed(cmd uintptr) {
	e.disp(cmd, "binembed", "embed", (uint32(e.hiddenN)+63)/64, 1, 1)
	e.bar(cmd)
}

func (e *hybridVKEngine) recordEmbedPrompt(cmd uintptr) {
	e.disp(cmd, "binembed_p", "embed_p", (uint32(e.hiddenN)+63)/64, 1, 1)
	e.bar(cmd)
}

func (e *hybridVKEngine) recordIncPos(cmd uintptr) {
	e.disp(cmd, "inc_pos", "inc_pos", 1, 1, 1)
	e.bar(cmd)
}

func (e *hybridVKEngine) runOneToken(id uint32, sample bool, writeTok bool) (uint32, error) {
	e.dispCount = 0
	if writeTok {
		e.writeU32(e.token, 0, id)
	}
	e.writeStep(uint32(e.pos), 0)
	err := e.dev.submitDispatches(func(cmd uintptr) {
		e.recordEmbed(cmd)
		e.recordLayers(cmd)
		if sample {
			e.recordSample(cmd)
		} else {
			e.recordIncPos(cmd)
		}
	})
	if err != nil {
		return 0, err
	}
	noteHybridStats(e.dispCount, true)
	e.pos++
	if !sample {
		return 0, nil
	}
	toks, err := e.readHistN(1)
	if err != nil {
		return 0, err
	}
	return toks[0], nil
}

func (e *hybridVKEngine) readHistN(n int) ([]uint32, error) {
	if n < 1 {
		return nil, fmt.Errorf("hist n < 1")
	}
	if e.histBuf.mapped == nil {
		return nil, fmt.Errorf("hist buffer unmapped")
	}
	raw := unsafe.Slice((*byte)(e.histBuf.mapped), n*4)
	out := make([]uint32, n)
	for i := 0; i < n; i++ {
		out[i] = binary.LittleEndian.Uint32(raw[i*4 : i*4+4])
	}
	return out, nil
}

func (e *hybridVKEngine) stepTokenSample(id uint32) (uint32, error) {
	return e.runOneToken(id, true, true)
}

func (e *hybridVKEngine) appendTokens(ids []uint32) ([]float32, error) {
	if len(ids) == 0 {
		return nil, fmt.Errorf("fusedgpu: empty ids")
	}
	logits := make([]float32, e.vocabN)
	for i, id := range ids {
		wantLogits := i == len(ids)-1
		e.writeU32(e.token, 0, id)
		e.writeStep(uint32(e.pos), 0)
		e.dispCount = 0
		err := e.dev.submitDispatches(func(cmd uintptr) {
			e.recordEmbed(cmd)
			e.recordLayers(cmd)
			if wantLogits {
				e.dispatchLM(cmd)
			}
			e.recordIncPos(cmd)
		})
		if err != nil {
			return nil, err
		}
		if wantLogits {
			if e.logits.mapped != nil {
				raw := unsafe.Slice((*byte)(e.logits.mapped), len(logits)*4)
				for j := range logits {
					logits[j] = math.Float32frombits(binary.LittleEndian.Uint32(raw[j*4 : j*4+4]))
				}
			}
		}
		e.pos++
	}
	return logits, nil
}

// prefillSample runs the prompt. Turnip A710 device-lost on multi-token command
// buffers (same as wgpu gpu_multi_fuse), so one token → one submit.
func (e *hybridVKEngine) prefillSample(ids []uint32) (uint32, error) {
	if len(ids) == 0 {
		return 0, fmt.Errorf("fusedgpu: empty ids")
	}
	if len(ids) > e.maxSeq {
		return 0, fmt.Errorf("fusedgpu: prompt len %d > maxSeq %d", len(ids), e.maxSeq)
	}
	e.pos = 0
	if len(ids) == 1 {
		return e.stepTokenSample(ids[0])
	}
	for i := 0; i < len(ids)-1; i++ {
		if _, err := e.runOneToken(ids[i], false, true); err != nil {
			return 0, fmt.Errorf("prefill step %d: %w", i, err)
		}
	}
	return e.stepTokenSample(ids[len(ids)-1])
}

func (e *hybridVKEngine) decodeChunkSample(k int) ([]uint32, error) {
	if k < 1 {
		return nil, fmt.Errorf("fusedgpu: decode chunk k < 1")
	}
	if e.pos+k > e.maxSeq {
		return nil, fmt.Errorf("fusedgpu: decode would exceed maxSeq %d", e.maxSeq)
	}
	out := make([]uint32, 0, k)
	// One decode step per submit — packing k>1 trips Turnip device-lost (-4).
	for i := 0; i < k; i++ {
		tok, err := e.runOneToken(0, true, false)
		if err != nil {
			return nil, fmt.Errorf("decode step %d/%d: %w", i+1, k, err)
		}
		out = append(out, tok)
	}
	return out, nil
}

func (e *hybridVKEngine) resetState() error {
	e.pos = 0
	e.writeStep(0, 0)
	return e.dev.submitDispatches(func(cmd uintptr) {
		p := e.pipe["zero"]
		for i := range e.blocks {
			b := &e.blocks[i]
			if b.kCache == nil {
				continue
			}
			n := uint32(b.numKVHeads * e.maxSeq * b.headDim)
			u := e.uni(fmt.Sprintf("clr%d_kc", i), packU32(n, 0, 0, 0))
			set, _ := e.dev.allocDescSet(p.setLayout)
			e.dev.writeDescriptorSet(set, []*vkBuffer{u, b.kCache}, nil, p.types)
			e.dev.bindDispatch(cmd, p.pipe, p.layout, set, (n+63)/64, 1, 1)
			u2 := e.uni(fmt.Sprintf("clr%d_vc", i), packU32(n, 0, 0, 0))
			set2, _ := e.dev.allocDescSet(p.setLayout)
			e.dev.writeDescriptorSet(set2, []*vkBuffer{u2, b.vCache}, nil, p.types)
			e.dev.bindDispatch(cmd, p.pipe, p.layout, set2, (n+63)/64, 1, 1)
		}
	})
}

func (e *hybridVKEngine) release() {
	if e == nil {
		return
	}
	for _, p := range e.pipe {
		e.dev.destroyPipelineBundle(p)
	}
	e.pipe = nil
	for _, b := range e.owned {
		b.destroy()
	}
	e.owned = nil
	e.blocks = nil
	if e.dev != nil {
		e.dev.destroy()
		e.dev = nil
	}
	e.spec = nil
	runtime.GC()
}
