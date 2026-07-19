package mosstts

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/openfluke/welvet/model/hf"
)

// Model is MossTTSNanoForCausalLM (global + local GPT2 + audio embeddings/heads).
type Model struct {
	Cfg    *Config
	Global *GPT2Model
	Local  *GPT2Model
	// Audio emb / heads: tied — store emb [nvq][codebook*hidden], heads share same storage.
	AudioEmb [][]float32
	TextHead Linear // tied to Global.WTE

	fuseGlobal *gpt2Fuse
	fuseLocal  *gpt2Fuse
	useFuseGPU bool
}

// LoadAR loads converted model.safetensors from a Welvet TTS snapshot.
func LoadAR(snapshotDir string) (*Model, error) {
	cfg, err := LoadConfig(snapshotDir)
	if err != nil {
		return nil, err
	}
	stPath := filepath.Join(snapshotDir, "model.safetensors")
	if _, err := os.Stat(stPath); err != nil {
		return nil, fmt.Errorf("mosstts: need model.safetensors (run octo/tools/moss_convert/convert.py): %w", err)
	}
	index, err := hf.BuildTensorIndex(stPath)
	if err != nil {
		return nil, err
	}
	get := func(name string) ([]float32, error) {
		return hf.LoadF16Vector(stPath, index, name)
	}
	// Prefer exact names; also try without prefix.
	getAny := func(names ...string) ([]float32, error) {
		var last error
		for _, n := range names {
			v, err := get(n)
			if err == nil {
				return v, nil
			}
			last = err
		}
		return nil, last
	}

	h := cfg.GPT2.NEmbd
	gBlocks := make([]gpt2Block, cfg.GPT2.NLayer)
	for i := 0; i < cfg.GPT2.NLayer; i++ {
		b, err := loadGPT2Block(getAny, "transformer", i, cfg)
		if err != nil {
			return nil, fmt.Errorf("global block %d: %w", i, err)
		}
		gBlocks[i] = b
	}
	wte, err := getAny("transformer.wte.weight")
	if err != nil {
		return nil, fmt.Errorf("wte: %w", err)
	}
	lnfW, err := getAny("transformer.ln_f.weight")
	if err != nil {
		return nil, err
	}
	lnfB, err := getAny("transformer.ln_f.bias")
	if err != nil {
		return nil, err
	}
	global := &GPT2Model{
		WTE: wte, Vocab: cfg.VocabSize, Hidden: h, Blocks: gBlocks,
		LNFW: lnfW, LNFB: lnfB, Eps: cfg.GPT2.LayerNormEps, HasWTE: true,
	}

	nLocal := cfg.LocalTransformerLayers
	lBlocks := make([]gpt2Block, nLocal)
	for i := 0; i < nLocal; i++ {
		b, err := loadGPT2Block(getAny, "local_transformer", i, cfg)
		if err != nil {
			return nil, fmt.Errorf("local block %d: %w", i, err)
		}
		lBlocks[i] = b
	}
	llnfW, err := getAny("local_transformer.ln_f.weight")
	if err != nil {
		return nil, err
	}
	llnfB, err := getAny("local_transformer.ln_f.bias")
	if err != nil {
		return nil, err
	}
	local := &GPT2Model{
		Hidden: h, Blocks: lBlocks, LNFW: llnfW, LNFB: llnfB,
		Eps: cfg.GPT2.LayerNormEps, HasWTE: false,
	}

	audioEmb := make([][]float32, cfg.NVQ)
	for i := 0; i < cfg.NVQ; i++ {
		emb, err := getAny(fmt.Sprintf("audio_embeddings.%d.weight", i))
		if err != nil {
			return nil, fmt.Errorf("audio_embeddings.%d: %w", i, err)
		}
		audioEmb[i] = emb
	}

	m := &Model{
		Cfg: cfg, Global: global, Local: local, AudioEmb: audioEmb,
		TextHead: Linear{Out: cfg.VocabSize, In: h, W: wte},
	}
	return m, nil
}

func loadGPT2Block(get func(...string) ([]float32, error), prefix string, i int, cfg *Config) (gpt2Block, error) {
	h := cfg.GPT2.NEmbd
	heads := cfg.GPT2.NHead
	inner := cfg.GPT2.NInner
	p := fmt.Sprintf("%s.h.%d", prefix, i)
	ln1w, err := get(p + ".ln_1.weight")
	if err != nil {
		return gpt2Block{}, err
	}
	ln1b, err := get(p + ".ln_1.bias")
	if err != nil {
		return gpt2Block{}, err
	}
	ln2w, err := get(p + ".ln_2.weight")
	if err != nil {
		return gpt2Block{}, err
	}
	ln2b, err := get(p + ".ln_2.bias")
	if err != nil {
		return gpt2Block{}, err
	}
	cAttnW, err := get(p + ".attn.c_attn.weight")
	if err != nil {
		return gpt2Block{}, err
	}
	cAttnB, err := get(p + ".attn.c_attn.bias")
	if err != nil {
		return gpt2Block{}, err
	}
	cProjW, err := get(p + ".attn.c_proj.weight")
	if err != nil {
		return gpt2Block{}, err
	}
	cProjB, err := get(p + ".attn.c_proj.bias")
	if err != nil {
		return gpt2Block{}, err
	}
	fcInW, err := get(p+".mlp.fc_in.weight", p+".mlp.c_fc.weight")
	if err != nil {
		return gpt2Block{}, err
	}
	fcInB, err := get(p+".mlp.fc_in.bias", p+".mlp.c_fc.bias")
	if err != nil {
		return gpt2Block{}, err
	}
	fcOutW, err := get(p+".mlp.fc_out.weight", p+".mlp.c_proj.weight")
	if err != nil {
		return gpt2Block{}, err
	}
	fcOutB, err := get(p+".mlp.fc_out.bias", p+".mlp.c_proj.bias")
	if err != nil {
		return gpt2Block{}, err
	}
	// PyTorch Linear weight is [out, in]; our Linear expects same.
	return gpt2Block{
		LN1W: ln1w, LN1B: ln1b, LN2W: ln2w, LN2B: ln2b,
		Attn: gpt2Attn{
			NumHeads: heads, HeadDim: h / heads, Hidden: h,
			CAttn: Linear{Out: 3 * h, In: h, W: cAttnW, B: cAttnB},
			CProj: Linear{Out: h, In: h, W: cProjW, B: cProjB},
			Scale: cfg.GPT2.ScaleAttnWeights, RopeBase: cfg.GPT2.RopeBase,
		},
		FcIn: Linear{Out: inner, In: h, W: fcInW, B: fcInB},
		FcOut: Linear{Out: h, In: inner, W: fcOutW, B: fcOutB},
		Eps: cfg.GPT2.LayerNormEps, Hidden: h,
	}, nil
}

func (m *Model) buildInputsEmbeds(rows [][]int) []float32 {
	// rows: [seq][1+nvq]
	cfg := m.Cfg
	h := cfg.HiddenSize
	seq := len(rows)
	out := make([]float32, seq*h)
	for t, row := range rows {
		tid := row[0]
		m.Global.embedToken(tid, out[t*h:(t+1)*h])
		for ch := 0; ch < cfg.NVQ; ch++ {
			aid := row[1+ch]
			if aid == cfg.AudioPadTokenID {
				continue
			}
			emb := m.AudioEmb[ch]
			cb := cfg.AudioCodebookSizes[ch]
			if aid < 0 || aid >= cb {
				continue
			}
			src := emb[aid*h : (aid+1)*h]
			dst := out[t*h : (t+1)*h]
			for i := 0; i < h; i++ {
				dst[i] += src[i]
			}
		}
	}
	return out
}

func (m *Model) audioLMHead(ch int, hidden, logits []float32) {
	emb := m.AudioEmb[ch]
	cb := m.Cfg.AudioCodebookSizes[ch]
	h := m.Cfg.HiddenSize
	l := Linear{Out: cb, In: h, W: emb, UseSIMD: m.TextHead.UseSIMD, UseGPU: m.TextHead.UseGPU}
	l.Forward(hidden, logits)
}

// SetFuse enables SIMD and/or resident GPT-2 GPU fuse for AR.
func (m *Model) SetFuse(simdOn, gpuOn bool) {
	if m == nil {
		return
	}
	m.Global.setFuse(simdOn, gpuOn)
	m.Local.setFuse(simdOn, gpuOn)
	m.TextHead.UseSIMD, m.TextHead.UseGPU = simdOn, gpuOn
	m.useFuseGPU = gpuOn
}

// SyncGPU builds resident GPT-2 fuse engines (one submit per decode token).
func (m *Model) SyncGPU() (int, error) {
	if m == nil {
		return 0, nil
	}
	m.CloseGPU()
	rope := m.Cfg.GPT2.RopeBase
	scale := m.Cfg.GPT2.ScaleAttnWeights
	maxSeq := m.Cfg.GPT2.NPositions
	if maxSeq <= 0 {
		maxSeq = 512
	}
	g, err := newGPT2Fuse(m.Global, maxSeq, rope, scale)
	if err != nil {
		return 0, err
	}
	l, err := newGPT2Fuse(m.Local, 64, rope, scale) // local seq is tiny (1+nvq)
	if err != nil {
		g.Close()
		return 0, err
	}
	m.fuseGlobal, m.fuseLocal = g, l
	n := len(m.Global.Blocks)*4 + len(m.Local.Blocks)*4
	return n, nil
}

// CloseGPU releases fuse engines.
func (m *Model) CloseGPU() {
	if m == nil {
		return
	}
	if m.fuseGlobal != nil {
		m.fuseGlobal.Close()
		m.fuseGlobal = nil
	}
	if m.fuseLocal != nil {
		m.fuseLocal.Close()
		m.fuseLocal = nil
	}
}

// FindAudioTokenizerDir locates MOSS-Audio-Tokenizer-Nano next to or inside the TTS snapshot.
func FindAudioTokenizerDir(ttsSnap string) (string, error) {
	candidates := []string{
		filepath.Join(ttsSnap, "audio_tokenizer"),
		filepath.Join(ttsSnap, "MOSS-Audio-Tokenizer-Nano"),
	}
	hub := filepath.Dir(filepath.Dir(filepath.Dir(ttsSnap))) // .../octo_hub
	candidates = append(candidates,
		filepath.Join(hub, "models--OpenMOSS-Team--MOSS-Audio-Tokenizer-Nano", "snapshots", "manual-download"),
	)
	for _, d := range candidates {
		if st, err := os.Stat(filepath.Join(d, "config.json")); err == nil && !st.IsDir() {
			if _, err := findSTFile(d); err == nil {
				return d, nil
			}
		}
	}
	return "", fmt.Errorf("audio tokenizer not found near %s (expected OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano snapshot)", ttsSnap)
}
