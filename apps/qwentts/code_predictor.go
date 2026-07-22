package qwentts

import (
	"fmt"
	"math"
	"path/filepath"

	"github.com/openfluke/welvet/model/hf"
)

// CodePredictor is the 5-layer Qwen3 MTP head that predicts codebooks 1..15
// from the talker hidden state and the first codebook (code0).
type CodePredictor struct {
	cfg        CodePredictorConfig
	numGroups  int // = talker num_code_groups - 1 (15)
	codebook   int // predictable code range (codebook_size, e.g. 2048)
	layers     []talkerLayer
	norm       []float32
	codecEmb   []*embedTable // 15 group embedding tables
	lmHead     []*Linear     // 15 group heads
	rope       *ropeCache
	kCache     [][]float32
	vCache     [][]float32
	kvLen      int
}

func loadCodePredictor(snapshotDir string, cfg CodePredictorConfig, numGroups int) (*CodePredictor, error) {
	path := filepath.Join(snapshotDir, "model.safetensors")
	idx, err := hf.BuildTensorIndex(path)
	if err != nil {
		return nil, fmt.Errorf("cp index: %w", err)
	}
	cp := &CodePredictor{cfg: cfg, numGroups: numGroups, codebook: cfg.VocabSize}
	if cp.norm, err = loadVec(path, idx, "talker.code_predictor.model.norm.weight"); err != nil {
		return nil, err
	}
	cp.layers = make([]talkerLayer, cfg.NumLayers)
	for i := 0; i < cfg.NumLayers; i++ {
		lp := fmt.Sprintf("talker.code_predictor.model.layers.%d", i)
		var l talkerLayer
		if l.inputLN, err = loadVec(path, idx, lp+".input_layernorm.weight"); err != nil {
			return nil, err
		}
		if l.postLN, err = loadVec(path, idx, lp+".post_attention_layernorm.weight"); err != nil {
			return nil, err
		}
		if l.q, err = loadLinear(path, idx, lp+".self_attn.q_proj", false); err != nil {
			return nil, err
		}
		if l.k, err = loadLinear(path, idx, lp+".self_attn.k_proj", false); err != nil {
			return nil, err
		}
		if l.v, err = loadLinear(path, idx, lp+".self_attn.v_proj", false); err != nil {
			return nil, err
		}
		if l.o, err = loadLinear(path, idx, lp+".self_attn.o_proj", false); err != nil {
			return nil, err
		}
		if l.qNorm, err = loadVec(path, idx, lp+".self_attn.q_norm.weight"); err != nil {
			return nil, err
		}
		if l.kNorm, err = loadVec(path, idx, lp+".self_attn.k_norm.weight"); err != nil {
			return nil, err
		}
		if l.gate, err = loadLinear(path, idx, lp+".mlp.gate_proj", false); err != nil {
			return nil, err
		}
		if l.up, err = loadLinear(path, idx, lp+".mlp.up_proj", false); err != nil {
			return nil, err
		}
		if l.down, err = loadLinear(path, idx, lp+".mlp.down_proj", false); err != nil {
			return nil, err
		}
		cp.layers[i] = l
	}
	cp.codecEmb = make([]*embedTable, numGroups)
	cp.lmHead = make([]*Linear, numGroups)
	for g := 0; g < numGroups; g++ {
		if cp.codecEmb[g], err = loadEmbed(path, idx, fmt.Sprintf("talker.code_predictor.model.codec_embedding.%d.weight", g)); err != nil {
			return nil, err
		}
		if cp.lmHead[g], err = loadLinear(path, idx, fmt.Sprintf("talker.code_predictor.lm_head.%d", g), false); err != nil {
			return nil, err
		}
	}
	maxPos := numGroups + 4
	cp.rope = newRopeCache(cfg.HeadDim, maxPos, cfg.RopeTheta)
	return cp, nil
}

func (cp *CodePredictor) resetCache() {
	cp.kCache = make([][]float32, cp.cfg.NumLayers)
	cp.vCache = make([][]float32, cp.cfg.NumLayers)
	cp.kvLen = 0
}

// forward runs nNew embeddings [nNew*H] through the CP stack (H = cp hidden),
// appends KV and returns post-norm hidden of the last position [H].
func (cp *CodePredictor) forward(embeds []float32, nNew int) []float32 {
	cfg := cp.cfg
	H := cfg.HiddenSize
	hd := cfg.HeadDim
	nh := cfg.NumHeads
	nkv := cfg.NumKVHeads
	qDim := nh * hd
	kvDim := nkv * hd
	rep := nh / nkv
	eps := cfg.RMSNormEps
	scale := float32(1 / math.Sqrt(float64(hd)))
	base := cp.kvLen

	h := make([]float32, nNew*H)
	copy(h, embeds[:nNew*H])
	xn := make([]float32, nNew*H)
	qAll := make([]float32, nNew*qDim)
	kAll := make([]float32, nNew*kvDim)
	vAll := make([]float32, nNew*kvDim)
	attn := make([]float32, nNew*qDim)
	tmp := make([]float32, nNew*H)
	gate := make([]float32, cfg.IntermediateSize)
	up := make([]float32, cfg.IntermediateSize)

	for li := 0; li < cfg.NumLayers; li++ {
		l := &cp.layers[li]
		for i := 0; i < nNew; i++ {
			rmsNormTo(xn[i*H:(i+1)*H], h[i*H:(i+1)*H], l.inputLN, H, eps)
		}
		l.q.forwardSeq(xn, qAll, nNew)
		l.k.forwardSeq(xn, kAll, nNew)
		l.v.forwardSeq(xn, vAll, nNew)
		for i := 0; i < nNew; i++ {
			pos := base + i
			q := qAll[i*qDim : (i+1)*qDim]
			k := kAll[i*kvDim : (i+1)*kvDim]
			for hh := 0; hh < nh; hh++ {
				rmsNorm(q[hh*hd:(hh+1)*hd], l.qNorm, hd, eps)
				cp.rope.applyRope(q[hh*hd:(hh+1)*hd], pos)
			}
			for hh := 0; hh < nkv; hh++ {
				rmsNorm(k[hh*hd:(hh+1)*hd], l.kNorm, hd, eps)
				cp.rope.applyRope(k[hh*hd:(hh+1)*hd], pos)
			}
		}
		cp.kCache[li] = append(cp.kCache[li], kAll[:nNew*kvDim]...)
		cp.vCache[li] = append(cp.vCache[li], vAll[:nNew*kvDim]...)
		kc := cp.kCache[li]
		vc := cp.vCache[li]
		total := base + nNew
		scores := make([]float32, total)
		for i := 0; i < nNew; i++ {
			qpos := base + i
			for hh := 0; hh < nh; hh++ {
				kvh := hh / rep
				qh := qAll[i*qDim+hh*hd : i*qDim+hh*hd+hd]
				n := qpos + 1
				for tpos := 0; tpos < n; tpos++ {
					kh := kc[tpos*kvDim+kvh*hd : tpos*kvDim+kvh*hd+hd]
					scores[tpos] = dotF32(qh, kh, hd) * scale
				}
				softmaxInPlace(scores, n)
				out := attn[i*qDim+hh*hd : i*qDim+hh*hd+hd]
				for d := 0; d < hd; d++ {
					out[d] = 0
				}
				for tpos := 0; tpos < n; tpos++ {
					w := scores[tpos]
					vh := vc[tpos*kvDim+kvh*hd : tpos*kvDim+kvh*hd+hd]
					for d := 0; d < hd; d++ {
						out[d] += w * vh[d]
					}
				}
			}
		}
		l.o.forwardSeq(attn, tmp, nNew)
		for j := 0; j < nNew*H; j++ {
			h[j] += tmp[j]
		}
		for i := 0; i < nNew; i++ {
			rmsNormTo(xn[i*H:(i+1)*H], h[i*H:(i+1)*H], l.postLN, H, eps)
			l.gate.forward(xn[i*H:(i+1)*H], gate)
			l.up.forward(xn[i*H:(i+1)*H], up)
			siluMul(gate, up)
			l.down.forward(gate, tmp[i*H:(i+1)*H])
		}
		for j := 0; j < nNew*H; j++ {
			h[j] += tmp[j]
		}
	}
	cp.kvLen += nNew
	last := make([]float32, H)
	rmsNormTo(last, h[(nNew-1)*H:nNew*H], cp.norm, H, eps)
	return last
}

// groupEmbed writes code-predictor group-i embedding of code into dst[H].
func (cp *CodePredictor) groupEmbed(i, code int, dst []float32) {
	cp.codecEmb[i].row(code, dst)
}

// predict returns 15 codebook indices (codes 1..15) given the talker's
// post-norm hidden and code0. code0Embed is talker.codec_embedding(code0).
func (cp *CodePredictor) predict(pastHidden, code0Embed []float32, smp *sampler) []int {
	cp.resetCache()
	H := cp.cfg.HiddenSize
	prefill := make([]float32, 2*H)
	copy(prefill[:H], pastHidden[:H])
	copy(prefill[H:], code0Embed[:H])
	hidden := cp.forward(prefill, 2)

	codes := make([]int, cp.numGroups)
	logits := make([]float32, cp.codebook)
	// group 0 (codebook 1)
	cp.lmHead[0].forward(hidden, logits)
	codes[0] = smp.sampleCP(logits)
	emb := make([]float32, H)
	for g := 1; g < cp.numGroups; g++ {
		cp.codecEmb[g-1].row(codes[g-1], emb)
		hidden = cp.forward(emb, 1)
		cp.lmHead[g].forward(hidden, logits)
		codes[g] = smp.sampleCP(logits)
	}
	return codes
}

// SetFuse enables SIMD on code-predictor linears.
// gpuOn sticky GEMV is intentionally unused during talker GPU fuse — per-mat
// WebGPU syncs (~100/frame) make AR slower than host SIMD.
func (cp *CodePredictor) SetFuse(simdOn, gpuOn bool) {
	if cp == nil {
		return
	}
	_ = gpuOn
	for i := range cp.layers {
		l := &cp.layers[i]
		setLinearFuseMany(simdOn, false, l.q, l.k, l.v, l.o, l.gate, l.up, l.down)
	}
	for _, h := range cp.lmHead {
		setLinearFuse(h, simdOn, false)
	}
}

// WarmGPU uploads large code-predictor weights.
// Returns early with IsF32VRAMFull when the sticky soft-cap is hit.
func (cp *CodePredictor) WarmGPU() (int, error) {
	if cp == nil {
		return 0, nil
	}
	n := 0
	warm := func(l *Linear) error {
		if l == nil || !l.UseGPU {
			return nil
		}
		if err := warmLinearGPU(l); err != nil {
			return err
		}
		n++
		return nil
	}
	for i := range cp.layers {
		l := &cp.layers[i]
		for _, lin := range []*Linear{l.q, l.k, l.v, l.o, l.gate, l.up, l.down} {
			if err := warm(lin); err != nil {
				return n, err
			}
		}
	}
	for _, h := range cp.lmHead {
		if err := warm(h); err != nil {
			return n, err
		}
	}
	return n, nil
}
