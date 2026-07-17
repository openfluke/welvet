package flux2

import (
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/openfluke/welvet/hf"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/tokenizer"
	"github.com/openfluke/welvet/transformer"
)

// Qwen3KleinConfig holds Qwen3-4B dims used by Flux2Klein text encoding.
type Qwen3KleinConfig struct {
	HiddenSize       int
	NumLayers        int
	NumHeads         int
	NumKVHeads       int
	HeadDim          int
	IntermediateSize int
	RopeTheta        float64
	VocabSize        int
	RMSNormEps       float64
	TieWordEmbeddings bool
}

// DefaultQwen3KleinConfig returns Qwen3-4B defaults for text_encoder-mlx-4bit.
func DefaultQwen3KleinConfig() Qwen3KleinConfig {
	return Qwen3KleinConfig{
		HiddenSize:        2560,
		NumLayers:         36,
		NumHeads:          32,
		NumKVHeads:        8,
		HeadDim:           128,
		IntermediateSize:  9728,
		RopeTheta:         1e6,
		VocabSize:         151936,
		RMSNormEps:        1e-6,
		TieWordEmbeddings: true,
	}
}

type qwen3Layer struct {
	InputLN, PostAttnLN []float32
	Q, K, V, O          *Linear
	Gate, Up, Down      *Linear
	QNorm, KNorm        []float32
}

// Qwen3TextEncoder is a Qwen3 decoder used only for Klein prompt embeds.
type Qwen3TextEncoder struct {
	Cfg       Qwen3KleinConfig
	Embed     *quant.Blob
	Layers    []qwen3Layer
	FinalNorm []float32 // unused for Klein mid-layer stack, kept for completeness
	UseGPU    bool
}

// LoadQwen3TextEncoder loads Affine4 linears + BF16/F16 norms from text_encoder-mlx-4bit/.
func LoadQwen3TextEncoder(teDir string) (*Qwen3TextEncoder, error) {
	cfg := DefaultQwen3KleinConfig()
	stPath, err := findTextEncoderWeights(teDir)
	if err != nil {
		return nil, err
	}
	index, err := hf.BuildTensorIndex(stPath)
	if err != nil {
		return nil, err
	}
	prefix := "model"
	if _, ok := index["model.embed_tokens.weight"]; !ok {
		if _, ok := index["language_model.model.embed_tokens.weight"]; ok {
			prefix = "language_model.model"
		} else {
			return nil, fmt.Errorf("LoadQwen3TextEncoder: no embed_tokens in %s", filepath.Base(stPath))
		}
	}

	emb, err := hf.LoadMLXAffineMatrix(stPath, index, prefix+".embed_tokens", 4, quant.AffineG64Group)
	if err != nil {
		return nil, fmt.Errorf("embed: %w", err)
	}
	cfg.VocabSize = emb.Rows
	if emb.Cols != cfg.HiddenSize {
		cfg.HiddenSize = emb.Cols
	}

	finalNorm, err := hf.LoadF16Vector(stPath, index, prefix+".norm.weight")
	if err != nil {
		// Some exports omit final norm when only mid-layers are needed; tolerate missing.
		finalNorm = nil
	}

	layers := make([]qwen3Layer, cfg.NumLayers)
	for i := 0; i < cfg.NumLayers; i++ {
		lp := fmt.Sprintf("%s.layers.%d", prefix, i)
		layer, err := loadQwen3Layer(stPath, index, lp)
		if err != nil {
			return nil, fmt.Errorf("layer %d: %w", i, err)
		}
		layers[i] = layer
	}

	return &Qwen3TextEncoder{
		Cfg:       cfg,
		Embed:     emb,
		Layers:    layers,
		FinalNorm: finalNorm,
	}, nil
}

func findTextEncoderWeights(teDir string) (string, error) {
	candidates := []string{
		filepath.Join(teDir, "model.safetensors"),
		filepath.Join(teDir, "model-00001-of-00001.safetensors"),
	}
	for _, p := range candidates {
		if st, err := os.Stat(p); err == nil && !st.IsDir() && st.Size() > 0 {
			return p, nil
		}
	}
	// Sharded: single-shard glob only for now.
	if sts, _ := filepath.Glob(filepath.Join(teDir, "model-*.safetensors")); len(sts) == 1 {
		return sts[0], nil
	} else if len(sts) > 1 {
		return "", fmt.Errorf("sharded text encoder not supported yet (%d shards under %s)", len(sts), teDir)
	}
	return "", fmt.Errorf("no model.safetensors under %s", teDir)
}

func loadQwen3Layer(stPath string, index map[string]hf.TensorInfo, lp string) (qwen3Layer, error) {
	var out qwen3Layer
	var err error
	if out.InputLN, err = hf.LoadF16Vector(stPath, index, lp+".input_layernorm.weight"); err != nil {
		return out, err
	}
	if out.PostAttnLN, err = hf.LoadF16Vector(stPath, index, lp+".post_attention_layernorm.weight"); err != nil {
		return out, err
	}
	loadLin := func(name string) (*Linear, error) {
		b, err := hf.LoadMLXAffineMatrix(stPath, index, lp+"."+name, 4, quant.AffineG64Group)
		if err != nil {
			return nil, err
		}
		return NewBlobLinear(b, nil, name)
	}
	if out.Q, err = loadLin("self_attn.q_proj"); err != nil {
		return out, err
	}
	if out.K, err = loadLin("self_attn.k_proj"); err != nil {
		return out, err
	}
	if out.V, err = loadLin("self_attn.v_proj"); err != nil {
		return out, err
	}
	if out.O, err = loadLin("self_attn.o_proj"); err != nil {
		return out, err
	}
	if out.Gate, err = loadLin("mlp.gate_proj"); err != nil {
		return out, err
	}
	if out.Up, err = loadLin("mlp.up_proj"); err != nil {
		return out, err
	}
	if out.Down, err = loadLin("mlp.down_proj"); err != nil {
		return out, err
	}
	if out.QNorm, err = hf.LoadF16Vector(stPath, index, lp+".self_attn.q_norm.weight"); err != nil {
		return out, err
	}
	if out.KNorm, err = hf.LoadF16Vector(stPath, index, lp+".self_attn.k_norm.weight"); err != nil {
		return out, err
	}
	return out, nil
}

// EncodeHiddenStates runs the decoder and returns stacked Klein embeds [seq*7680].
//
// HF/Diffusers caveat: output.hidden_states[0] = embeddings; hidden_states[k] =
// activations after completing layer index (k-1). Diffusers Klein indexes
// hidden_states[9], [18], [27] — i.e. after layers 8, 17, 26 (0-based).
func (m *Qwen3TextEncoder) EncodeHiddenStates(ids []uint32, layerIdxs []int) ([]float32, int, error) {
	if m == nil || m.Embed == nil {
		return nil, 0, fmt.Errorf("EncodeHiddenStates: nil encoder")
	}
	seq := len(ids)
	if seq == 0 {
		return nil, 0, fmt.Errorf("EncodeHiddenStates: empty ids")
	}
	if len(layerIdxs) == 0 {
		layerIdxs = DefaultKleinHiddenLayers
	}
	want := make(map[int]bool, len(layerIdxs))
	maxHS := 0
	for _, k := range layerIdxs {
		want[k] = true
		if k > maxHS {
			maxHS = k
		}
	}
	// hidden_states[k] requires completing layer (k-1); skip later layers.
	maxLayer := maxHS - 1
	if maxLayer < 0 {
		maxLayer = -1
	}
	if maxLayer >= len(m.Layers) {
		maxLayer = len(m.Layers) - 1
	}

	hSize := m.Cfg.HiddenSize
	h := make([]float32, seq*hSize)
	for t, id := range ids {
		if int(id) >= m.Embed.Rows {
			return nil, 0, fmt.Errorf("token %d OOB vocab %d", id, m.Embed.Rows)
		}
		if err := quant.DecodeRow(m.Embed, int(id), h[t*hSize:(t+1)*hSize]); err != nil {
			return nil, 0, err
		}
	}

	collected := make(map[int][]float32, len(layerIdxs))
	// hs index 0 = embeddings (not collected by default Klein idxs).
	if want[0] {
		cp := make([]float32, len(h))
		copy(cp, h)
		collected[0] = cp
	}

	normScratch := make([]float32, seq*hSize)
	mix := make([]float32, seq*hSize)
	for i := 0; i <= maxLayer; i++ {
		fmt.Printf("    text-enc layer %d/%d…\n", i+1, maxLayer+1)
		layer := &m.Layers[i]
		copy(normScratch, h)
		RMSNormSeq(normScratch, layer.InputLN, seq, hSize, m.Cfg.RMSNormEps)
		if err := qwen3AttnPrefill(layer, normScratch, mix, seq, m.Cfg); err != nil {
			return nil, 0, fmt.Errorf("layer %d attn: %w", i, err)
		}
		for j := range h {
			h[j] += mix[j]
		}
		copy(normScratch, h)
		RMSNormSeq(normScratch, layer.PostAttnLN, seq, hSize, m.Cfg.RMSNormEps)
		if err := qwen3MLP(layer, normScratch, mix, seq); err != nil {
			return nil, 0, fmt.Errorf("layer %d mlp: %w", i, err)
		}
		for j := range h {
			h[j] += mix[j]
		}
		hsIdx := i + 1
		if want[hsIdx] {
			cp := make([]float32, len(h))
			copy(cp, h)
			collected[hsIdx] = cp
		}
	}
	ordered := make([][]float32, 0, len(layerIdxs))
	for _, k := range layerIdxs {
		hs, ok := collected[k]
		if !ok {
			return nil, 0, fmt.Errorf("EncodeHiddenStates: missing hidden_states[%d]", k)
		}
		ordered = append(ordered, hs)
	}

	// Stack as Diffusers: [3, S, H] then permute → [S, 3, H] → [S, 3H].
	joint := len(ordered) * hSize
	out := make([]float32, seq*joint)
	for t := 0; t < seq; t++ {
		for c, hs := range ordered {
			copy(out[t*joint+c*hSize:t*joint+(c+1)*hSize], hs[t*hSize:(t+1)*hSize])
		}
	}
	return out, seq, nil
}

func qwen3MLP(layer *qwen3Layer, x, y []float32, seq int) error {
	in := layer.Gate.In
	inter := layer.Gate.Out
	gate := make([]float32, inter)
	up := make([]float32, inter)
	fused := make([]float32, inter)
	downIn := make([]float32, inter)
	for s := 0; s < seq; s++ {
		xs := x[s*in : (s+1)*in]
		if err := layer.Gate.MatVec(xs, gate); err != nil {
			return err
		}
		if err := layer.Up.MatVec(xs, up); err != nil {
			return err
		}
		for i := 0; i < inter; i++ {
			// SiLU(gate) * up
			g := gate[i]
			fused[i] = (g / float32(1+math.Exp(float64(-g)))) * up[i]
		}
		copy(downIn, fused)
		if err := layer.Down.MatVec(downIn, y[s*layer.Down.Out:(s+1)*layer.Down.Out]); err != nil {
			return err
		}
	}
	return nil
}

func qwen3AttnPrefill(layer *qwen3Layer, x, y []float32, seq int, cfg Qwen3KleinConfig) error {
	hd, nh, nkv := cfg.HeadDim, cfg.NumHeads, cfg.NumKVHeads
	qDim := nh * hd
	kvDim := nkv * hd
	eps := cfg.RMSNormEps

	qAll := make([]float32, seq*qDim)
	kAll := make([]float32, seq*kvDim)
	vAll := make([]float32, seq*kvDim)
	if err := layer.Q.MatMulSeq(x, qAll, seq); err != nil {
		return err
	}
	if err := layer.K.MatMulSeq(x, kAll, seq); err != nil {
		return err
	}
	if err := layer.V.MatMulSeq(x, vAll, seq); err != nil {
		return err
	}

	// Per-head RMSNorm on Q/K, then RoPE (rotate_half) at each position.
	for s := 0; s < seq; s++ {
		q := qAll[s*qDim : (s+1)*qDim]
		k := kAll[s*kvDim : (s+1)*kvDim]
		for h := 0; h < nh; h++ {
			RMSNorm(q[h*hd:(h+1)*hd], layer.QNorm, hd, eps)
		}
		for h := 0; h < nkv; h++ {
			RMSNorm(k[h*hd:(h+1)*hd], layer.KNorm, hd, eps)
		}
		transformer.ApplyPartialRoPE(q, nh, hd, 1.0, cfg.RopeTheta, s)
		transformer.ApplyPartialRoPE(k, nkv, hd, 1.0, cfg.RopeTheta, s)
	}

	attnOut := make([]float32, seq*qDim)
	scale := float32(1 / math.Sqrt(float64(hd)))
	rep := nh / nkv
	scores := make([]float32, seq)
	for s := 0; s < seq; s++ {
		for h := 0; h < nh; h++ {
			kvH := h / rep
			qh := qAll[s*qDim+h*hd : s*qDim+h*hd+hd]
			var maxS float32 = -1e30
			for t := 0; t <= s; t++ {
				kh := kAll[t*kvDim+kvH*hd : t*kvDim+kvH*hd+hd]
				var dot float32
				for d := 0; d < hd; d++ {
					dot += qh[d] * kh[d]
				}
				sc := dot * scale
				scores[t] = sc
				if sc > maxS {
					maxS = sc
				}
			}
			var sum float32
			for t := 0; t <= s; t++ {
				scores[t] = float32(math.Exp(float64(scores[t] - maxS)))
				sum += scores[t]
			}
			inv := 1 / sum
			outH := attnOut[s*qDim+h*hd : s*qDim+h*hd+hd]
			for d := 0; d < hd; d++ {
				var acc float32
				for t := 0; t <= s; t++ {
					vh := vAll[t*kvDim+kvH*hd : t*kvDim+kvH*hd+hd]
					acc += scores[t] * inv * vh[d]
				}
				outH[d] = acc
			}
		}
	}
	return layer.O.MatMulSeq(attnOut, y, seq)
}

// EncodePromptQwen runs chat-template tokenize → Qwen3 mid-layer stack → [S, 7680].
func EncodePromptQwen(snapshotDir, prompt string, maxLen int) ([]float32, int, error) {
	if maxLen <= 0 {
		maxLen = DefaultMaxPromptTokens
	}
	teDir, err := FindTextEncoderDir(snapshotDir)
	if err != nil {
		return nil, 0, err
	}
	tokDir, err := FindTokenizerDir(snapshotDir)
	if err != nil {
		return nil, 0, err
	}
	tok, err := tokenizer.LoadTokenizer(filepath.Join(tokDir, "tokenizer.json"))
	if err != nil {
		return nil, 0, fmt.Errorf("load tokenizer: %w", err)
	}

	// Qwen3 chat template: user turn + assistant prefix + empty <think> (enable_thinking=False).
	templated := transformer.ChatML.BuildPromptNoThink(nil, "", prompt)
	ids32 := tok.Encode(templated, true)
	if len(ids32) > maxLen {
		ids32 = ids32[:maxLen]
	}
	if len(ids32) == 0 {
		return nil, 0, fmt.Errorf("EncodePromptQwen: empty token ids")
	}

	enc, err := LoadQwen3TextEncoder(teDir)
	if err != nil {
		return nil, 0, err
	}
	// Klein indexes hidden_states[27] → need layers 0..26 on GPU.
	if err := enc.SyncGPU(26); err != nil {
		return nil, 0, fmt.Errorf("text encoder SyncGPU: %w", err)
	}
	defer enc.CloseGPU()
	return enc.EncodeHiddenStates(ids32, DefaultKleinHiddenLayers)
}
