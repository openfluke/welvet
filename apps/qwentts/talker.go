package qwentts

import (
	"fmt"
	"math"
	"path/filepath"

	"github.com/openfluke/welvet/model/hf"
)

type talkerLayer struct {
	inputLN, postLN []float32
	q, k, v, o      *Linear
	qNorm, kNorm    []float32
	gate, up, down  *Linear
}

// Talker is the Qwen3 talker LLM with text/codec embeddings, text_projection,
// codec_head and a persistent KV cache for autoregressive frame generation.
type Talker struct {
	cfg TalkerConfig

	textEmbed *embedTable
	projFC1   *Linear
	projFC2   *Linear
	codecEmb  *embedTable
	codecHead *Linear
	norm      []float32
	layers    []talkerLayer
	rope      *ropeCache

	// KV cache (per layer flat [pos*kvDim]).
	kCache [][]float32
	vCache [][]float32
	kvLen  int

	// fuse is the resident FP32 GPU decode engine (one submit per token).
	// When non-nil it replaces the host layer stack; layer Linears keep their
	// SIMD flag only as a host fallback.
	fuse *talkerFuse
}

func loadTalker(snapshotDir string, cfg TalkerConfig) (*Talker, error) {
	path := filepath.Join(snapshotDir, "model.safetensors")
	idx, err := hf.BuildTensorIndex(path)
	if err != nil {
		return nil, fmt.Errorf("talker index: %w", err)
	}
	t := &Talker{cfg: cfg}
	if t.textEmbed, err = loadEmbed(path, idx, "talker.model.text_embedding.weight"); err != nil {
		return nil, err
	}
	if t.codecEmb, err = loadEmbed(path, idx, "talker.model.codec_embedding.weight"); err != nil {
		return nil, err
	}
	if t.projFC1, err = loadLinear(path, idx, "talker.text_projection.linear_fc1", true); err != nil {
		return nil, err
	}
	if t.projFC2, err = loadLinear(path, idx, "talker.text_projection.linear_fc2", true); err != nil {
		return nil, err
	}
	if t.codecHead, err = loadLinear(path, idx, "talker.codec_head", false); err != nil {
		return nil, err
	}
	if t.norm, err = loadVec(path, idx, "talker.model.norm.weight"); err != nil {
		return nil, err
	}
	t.layers = make([]talkerLayer, cfg.NumLayers)
	for i := 0; i < cfg.NumLayers; i++ {
		lp := fmt.Sprintf("talker.model.layers.%d", i)
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
		t.layers[i] = l
	}
	maxPos := cfg.MaxPos
	if maxPos < 4096 {
		maxPos = 4096
	}
	t.rope = newRopeCache(cfg.HeadDim, maxPos, cfg.RopeTheta)
	t.resetCache()
	return t, nil
}

func (t *Talker) resetCache() {
	t.kCache = make([][]float32, t.cfg.NumLayers)
	t.vCache = make([][]float32, t.cfg.NumLayers)
	t.kvLen = 0
	if t.fuse != nil {
		_ = t.fuse.ResetPos()
	}
}

// embedText returns text_projection(text_embedding(id)) into dst[hidden].
func (t *Talker) embedText(id int, dst []float32) {
	te := make([]float32, t.textEmbed.Dim)
	t.textEmbed.row(id, te)
	fc1 := make([]float32, t.projFC1.Out)
	t.projFC1.forward(te, fc1)
	siluMulInPlace(fc1)
	t.projFC2.forward(fc1, dst[:t.projFC2.Out])
}

// embedCodec returns codec_embedding(id) into dst[hidden].
func (t *Talker) embedCodec(id int, dst []float32) {
	t.codecEmb.row(id, dst)
}

// forward runs nNew token embeddings through the stack, appends to the KV
// cache and returns the post-final-norm hidden of the LAST position [hidden].
func (t *Talker) forward(embeds []float32, nNew int) []float32 {
	cfg := t.cfg
	H := cfg.HiddenSize

	// True GPU fuse: run each token through the resident one-submit decode.
	// Prefill (nNew>1) is processed sequentially so the GPU KV cache matches a
	// batched prefill exactly (causal attention with a growing cache).
	if t.fuse != nil {
		last := make([]float32, H)
		x := make([]float32, H)
		for i := 0; i < nNew; i++ {
			copy(x, embeds[i*H:(i+1)*H])
			out, err := t.fuse.DecodeStepFused(x)
			if err != nil {
				t.fuse = nil
				// Only the fresh-prefill case can safely restart on the host
				// (host KV cache is still empty). Otherwise best-effort return.
				if t.kvLen == 0 {
					return t.forward(embeds, nNew)
				}
				last := make([]float32, H)
				copy(last, x)
				return last
			}
			if i == nNew-1 {
				copy(last, out)
			}
		}
		t.kvLen += nNew
		return last
	}

	hd := cfg.HeadDim
	nh := cfg.NumHeads
	nkv := cfg.NumKVHeads
	qDim := nh * hd
	kvDim := nkv * hd
	rep := nh / nkv
	eps := cfg.RMSNormEps
	scale := float32(1 / math.Sqrt(float64(hd)))
	base := t.kvLen

	h := make([]float32, nNew*H)
	copy(h, embeds[:nNew*H])

	xn := make([]float32, nNew*H)
	qAll := make([]float32, nNew*qDim)
	kAll := make([]float32, nNew*kvDim)
	vAll := make([]float32, nNew*kvDim)
	attn := make([]float32, nNew*qDim)
	mlpOut := make([]float32, nNew*H)
	gate := make([]float32, cfg.IntermediateSize)
	up := make([]float32, cfg.IntermediateSize)

	for li := 0; li < cfg.NumLayers; li++ {
		l := &t.layers[li]
		// attention pre-norm + qkv
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
				t.rope.applyRope(q[hh*hd:(hh+1)*hd], pos)
			}
			for hh := 0; hh < nkv; hh++ {
				rmsNorm(k[hh*hd:(hh+1)*hd], l.kNorm, hd, eps)
				t.rope.applyRope(k[hh*hd:(hh+1)*hd], pos)
			}
		}
		// append to cache
		t.kCache[li] = append(t.kCache[li], kAll[:nNew*kvDim]...)
		t.vCache[li] = append(t.vCache[li], vAll[:nNew*kvDim]...)
		total := base + nNew
		kc := t.kCache[li]
		vc := t.vCache[li]
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
		// o_proj + residual
		l.o.forwardSeq(attn, mlpOut, nNew)
		for j := 0; j < nNew*H; j++ {
			h[j] += mlpOut[j]
		}
		// mlp
		for i := 0; i < nNew; i++ {
			rmsNormTo(xn[i*H:(i+1)*H], h[i*H:(i+1)*H], l.postLN, H, eps)
			l.gate.forward(xn[i*H:(i+1)*H], gate)
			l.up.forward(xn[i*H:(i+1)*H], up)
			siluMul(gate, up)
			l.down.forward(gate, mlpOut[i*H:(i+1)*H])
		}
		for j := 0; j < nNew*H; j++ {
			h[j] += mlpOut[j]
		}
	}
	t.kvLen += nNew

	last := make([]float32, H)
	rmsNormTo(last, h[(nNew-1)*H:nNew*H], t.norm, H, eps)
	return last
}

// codecLogits computes codec_head @ lastHidden into logits[codecVocab].
func (t *Talker) codecLogits(lastHidden, logits []float32) {
	t.codecHead.forward(lastHidden, logits)
}

// SetFuse configures the talker execution backend.
//
//   - gpuOn: build the resident FP32 GPU decode fuse (one submit + one readback
//     per token). Layer weights live only in the fuse. Heads (text_projection,
//     codec_head) stay on SIMD/host — sticky per-GEMV WebGPU beside the fuse
//     adds extra syncs every frame and erases the win (same rule as mosstts).
//   - !gpuOn: tear down the fuse; linears use SIMD (and optional sticky later).
func (t *Talker) SetFuse(simdOn, gpuOn bool) {
	if t == nil {
		return
	}
	// Heads + layers: SIMD only while the resident fuse owns the GPU.
	setLinearFuseMany(simdOn, false, t.projFC1, t.projFC2, t.codecHead)
	for i := range t.layers {
		l := &t.layers[i]
		setLinearFuseMany(simdOn, false, l.q, l.k, l.v, l.o, l.gate, l.up, l.down)
	}

	if gpuOn {
		if t.fuse != nil {
			t.fuse.Close()
			t.fuse = nil
		}
		maxSeq := t.cfg.MaxPos
		if maxSeq <= 0 || maxSeq > 2048 {
			maxSeq = 2048
		}
		f, err := newTalkerFuse(t, maxSeq)
		if err != nil {
			fmt.Printf("  qwen talker fuse init failed: %v (using SIMD host)\n", err)
			return
		}
		t.fuse = f
		return
	}
	t.CloseFuse()
}

// CloseFuse tears down the resident GPU decode fuse (if any).
func (t *Talker) CloseFuse() {
	if t == nil || t.fuse == nil {
		return
	}
	t.fuse.Close()
	t.fuse = nil
}

// FuseMaxSeq reports the fused decode context cap (0 when no fuse is active).
func (t *Talker) FuseMaxSeq() int {
	if t == nil || t.fuse == nil {
		return 0
	}
	return t.fuse.maxSeq
}

// WarmGPU uploads large talker weights to sticky VRAM.
// Soft-stops on F32 VRAM budget (caller treats IsF32VRAMFull as non-fatal).
func (t *Talker) WarmGPU() (int, error) {
	if t == nil {
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
	for _, l := range []*Linear{t.projFC1, t.projFC2, t.codecHead} {
		if err := warm(l); err != nil {
			return n, err
		}
	}
	// Skip layer mats when the resident decode fuse owns them (UseGPU=false).
	for i := range t.layers {
		l := &t.layers[i]
		for _, lin := range []*Linear{l.q, l.k, l.v, l.o, l.gate, l.up, l.down} {
			if err := warm(lin); err != nil {
				return n, err
			}
		}
	}
	return n, nil
}
