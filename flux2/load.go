package flux2

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/openfluke/welvet/hf"
)

// LoadTransformerFromMLX loads Flux2 weights from a Bonsai/MLX snapshot directory.
// Expects transformer-packed-mflux/{config.json, diffusion_pytorch_model.safetensors}.
func LoadTransformerFromMLX(snapshotDir string) (*Model, error) {
	cfg, err := LoadConfig(snapshotDir)
	if err != nil {
		return nil, err
	}
	stPath, err := findSafetensors(snapshotDir)
	if err != nil {
		return nil, err
	}
	index, err := hf.BuildTensorIndex(stPath)
	if err != nil {
		return nil, err
	}
	m := &Model{Cfg: cfg}
	dim := cfg.InnerDim()
	mlpH := cfg.MLPHiddenDim()
	headDim := cfg.AttentionHeadDim
	heads := cfg.NumAttentionHeads

	loadQ := func(base string) (*Linear, error) {
		return loadLinear(stPath, index, base, true)
	}
	loadD := func(base string) (*Linear, error) {
		return loadLinear(stPath, index, base, false)
	}

	if m.XEmbedder, err = loadD("x_embedder"); err != nil {
		return nil, err
	}
	if m.ContextEmbedder, err = loadD("context_embedder"); err != nil {
		return nil, err
	}
	if m.TimeLinear1, err = loadD("time_guidance_embed.timestep_embedder.linear_1"); err != nil {
		// try alternate naming
		if m.TimeLinear1, err = loadD("time_text_embed.timestep_embedder.linear_1"); err != nil {
			return nil, fmt.Errorf("time embed linear_1: %w", err)
		}
	}
	if m.TimeLinear2, err = loadD("time_guidance_embed.timestep_embedder.linear_2"); err != nil {
		if m.TimeLinear2, err = loadD("time_text_embed.timestep_embedder.linear_2"); err != nil {
			return nil, fmt.Errorf("time embed linear_2: %w", err)
		}
	}
	if m.DoubleModImg, err = loadD("double_stream_modulation_img.linear"); err != nil {
		return nil, err
	}
	if m.DoubleModTxt, err = loadD("double_stream_modulation_txt.linear"); err != nil {
		return nil, err
	}
	if m.SingleMod, err = loadD("single_stream_modulation.linear"); err != nil {
		return nil, err
	}
	if m.NormOutLinear, err = loadD("norm_out.linear"); err != nil {
		return nil, err
	}
	if m.ProjOut, err = loadD("proj_out"); err != nil {
		return nil, err
	}

	m.DoubleBlocks = make([]DoubleStreamBlock, cfg.NumLayers)
	for i := 0; i < cfg.NumLayers; i++ {
		p := fmt.Sprintf("transformer_blocks.%d", i)
		blk := DoubleStreamBlock{
			Heads: heads, HeadDim: headDim, Dim: dim, MLPHidden: mlpH, Eps: cfg.Eps,
		}
		if blk.ToQ, err = loadQ(p + ".attn.to_q"); err != nil {
			return nil, err
		}
		if blk.ToK, err = loadQ(p + ".attn.to_k"); err != nil {
			return nil, err
		}
		if blk.ToV, err = loadQ(p + ".attn.to_v"); err != nil {
			return nil, err
		}
		if blk.AddQ, err = loadQ(p + ".attn.add_q_proj"); err != nil {
			return nil, err
		}
		if blk.AddK, err = loadQ(p + ".attn.add_k_proj"); err != nil {
			return nil, err
		}
		if blk.AddV, err = loadQ(p + ".attn.add_v_proj"); err != nil {
			return nil, err
		}
		if blk.ToOut, err = loadQ(p + ".attn.to_out.0"); err != nil {
			return nil, err
		}
		if blk.ToAddOut, err = loadQ(p + ".attn.to_add_out"); err != nil {
			return nil, err
		}
		if blk.FFIn, err = loadQ(p + ".ff.linear_in"); err != nil {
			return nil, err
		}
		if blk.FFOut, err = loadQ(p + ".ff.linear_out"); err != nil {
			return nil, err
		}
		if blk.FFContextIn, err = loadQ(p + ".ff_context.linear_in"); err != nil {
			return nil, err
		}
		if blk.FFContextOut, err = loadQ(p + ".ff_context.linear_out"); err != nil {
			return nil, err
		}
		if blk.NormQ, err = hf.LoadF16Vector(stPath, index, p+".attn.norm_q.weight"); err != nil {
			return nil, err
		}
		if blk.NormK, err = hf.LoadF16Vector(stPath, index, p+".attn.norm_k.weight"); err != nil {
			return nil, err
		}
		if blk.NormAddedQ, err = hf.LoadF16Vector(stPath, index, p+".attn.norm_added_q.weight"); err != nil {
			return nil, err
		}
		if blk.NormAddedK, err = hf.LoadF16Vector(stPath, index, p+".attn.norm_added_k.weight"); err != nil {
			return nil, err
		}
		m.DoubleBlocks[i] = blk
	}

	m.SingleBlocks = make([]SingleStreamBlock, cfg.NumSingleLayers)
	for i := 0; i < cfg.NumSingleLayers; i++ {
		p := fmt.Sprintf("single_transformer_blocks.%d", i)
		blk := SingleStreamBlock{
			Heads: heads, HeadDim: headDim, Dim: dim, MLPHidden: mlpH, Eps: cfg.Eps,
		}
		if blk.ToQKVMLP, err = loadQ(p + ".attn.to_qkv_mlp_proj"); err != nil {
			return nil, err
		}
		if blk.ToOut, err = loadQ(p + ".attn.to_out"); err != nil {
			return nil, err
		}
		if blk.NormQ, err = hf.LoadF16Vector(stPath, index, p+".attn.norm_q.weight"); err != nil {
			return nil, err
		}
		if blk.NormK, err = hf.LoadF16Vector(stPath, index, p+".attn.norm_k.weight"); err != nil {
			return nil, err
		}
		m.SingleBlocks[i] = blk
	}
	return m, nil
}

func findSafetensors(snapshotDir string) (string, error) {
	candidates := []string{
		filepath.Join(snapshotDir, "transformer-packed-mflux", "diffusion_pytorch_model.safetensors"),
		filepath.Join(snapshotDir, "transformer", "diffusion_pytorch_model.safetensors"),
		filepath.Join(snapshotDir, "diffusion_pytorch_model.safetensors"),
	}
	for _, p := range candidates {
		if st, err := os.Stat(p); err == nil && !st.IsDir() {
			return p, nil
		}
	}
	return "", fmt.Errorf("LoadTransformerFromMLX: no diffusion_pytorch_model.safetensors under %s", snapshotDir)
}

// loadLinear loads a quantized BinaryG128 matrix when scales exist, else dense BF16/F16.
func loadLinear(stPath string, index map[string]hf.TensorInfo, base string, preferQuant bool) (*Linear, error) {
	if preferQuant {
		if _, ok := index[base+".scales"]; ok {
			blob, err := hf.LoadMLX1BitMatrix(stPath, index, base)
			if err != nil {
				return nil, fmt.Errorf("quant %s: %w", base, err)
			}
			return NewBlobLinear(blob, nil, base)
		}
	}
	// dense skip patterns / fallback
	wName := base + ".weight"
	ti, ok := index[wName]
	if !ok {
		return nil, fmt.Errorf("missing %s", wName)
	}
	if len(ti.Shape) != 2 {
		return nil, fmt.Errorf("%s: expected 2D weight, got %v", wName, ti.Shape)
	}
	w, err := hf.LoadF16Vector(stPath, index, wName)
	if err != nil {
		return nil, err
	}
	out, in := ti.Shape[0], ti.Shape[1]
	var bias []float32
	if _, ok := index[base+".bias"]; ok {
		bias, err = hf.LoadF16Vector(stPath, index, base+".bias")
		if err != nil {
			return nil, err
		}
	}
	return NewDenseLinear(out, in, w, bias, base)
}

// IsSkipDense reports tensors that stay BF16 dense (not BinaryG128).
func IsSkipDense(name string) bool {
	skip := []string{
		"proj_out", "x_embedder", "context_embedder",
		"time_", "norm_out", "modulation",
	}
	for _, s := range skip {
		if strings.Contains(name, s) {
			return true
		}
	}
	return false
}
