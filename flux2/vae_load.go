package flux2

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/openfluke/welvet/hf"
)

// LoadVAEFromDir loads AutoencoderKLFlux2 decoder weights from snapshotDir/vae/.
// Expects vae/config.json and vae/diffusion_pytorch_model.safetensors (BF16/F16 dense).
// Encoder / quant_conv tensors are ignored (decode-only).
func LoadVAEFromDir(snapshotDir string) (*AutoencoderKLFlux2, error) {
	vaeDir := filepath.Join(snapshotDir, "vae")
	cfgPath := filepath.Join(vaeDir, "config.json")
	stPath := filepath.Join(vaeDir, "diffusion_pytorch_model.safetensors")
	if _, err := os.Stat(cfgPath); err != nil {
		return nil, fmt.Errorf("LoadVAEFromDir: %w", err)
	}
	if _, err := os.Stat(stPath); err != nil {
		return nil, fmt.Errorf("LoadVAEFromDir: %w", err)
	}

	v := NewVAEStub()
	if err := applyVAEConfigJSON(v, cfgPath); err != nil {
		return nil, err
	}
	index, err := hf.BuildTensorIndex(stPath)
	if err != nil {
		return nil, err
	}

	// BatchNorm running stats (used by Klein denorm before unpatchify).
	if v.BNMean, err = hf.LoadF16Vector(stPath, index, "bn.running_mean"); err != nil {
		return nil, fmt.Errorf("bn.running_mean: %w", err)
	}
	if v.BNVar, err = hf.LoadF16Vector(stPath, index, "bn.running_var"); err != nil {
		return nil, fmt.Errorf("bn.running_var: %w", err)
	}

	if v.PostQuant, err = loadConv2d(stPath, index, "post_quant_conv", 0); err != nil {
		return nil, err
	}
	if v.ConvIn, err = loadConv2d(stPath, index, "decoder.conv_in", 1); err != nil {
		return nil, err
	}
	if v.NormOut, err = loadGroupNorm(stPath, index, "decoder.conv_norm_out", v.NormNumGroups, 1e-6); err != nil {
		return nil, err
	}
	if v.ConvOut, err = loadConv2d(stPath, index, "decoder.conv_out", 1); err != nil {
		return nil, err
	}

	midCh := v.BlockOutChannels[len(v.BlockOutChannels)-1]
	mid := &vaeMidBlock{}
	if mid.Res0, err = loadResnet(stPath, index, "decoder.mid_block.resnets.0", v.NormNumGroups); err != nil {
		return nil, err
	}
	if mid.Res1, err = loadResnet(stPath, index, "decoder.mid_block.resnets.1", v.NormNumGroups); err != nil {
		return nil, err
	}
	if mid.Attn, err = loadVAEAttention(stPath, index, "decoder.mid_block.attentions.0", midCh, v.NormNumGroups); err != nil {
		return nil, err
	}
	v.Mid = mid

	// Up blocks: reversed block_out_channels, layers_per_block+1 resnets each.
	rev := reverseInts(v.BlockOutChannels)
	v.UpBlocks = make([]*vaeUpBlock, len(rev))
	prev := rev[0]
	numRes := v.LayersPerBlock + 1
	for i := range rev {
		outCh := rev[i]
		blk := &vaeUpBlock{}
		blk.Resnets = make([]*resnetBlock2D, numRes)
		for j := 0; j < numRes; j++ {
			prefix := fmt.Sprintf("decoder.up_blocks.%d.resnets.%d", i, j)
			blk.Resnets[j], err = loadResnet(stPath, index, prefix, v.NormNumGroups)
			if err != nil {
				return nil, err
			}
			_ = prev // first resnet may change channels via shortcut
		}
		isFinal := i == len(rev)-1
		if !isFinal {
			upPrefix := fmt.Sprintf("decoder.up_blocks.%d.upsamplers.0.conv", i)
			blk.Upsampler, err = loadConv2d(stPath, index, upPrefix, 1)
			if err != nil {
				return nil, err
			}
		}
		v.UpBlocks[i] = blk
		prev = outCh
	}

	v.Loaded = true
	return v, nil
}

func applyVAEConfigJSON(v *AutoencoderKLFlux2, path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		return fmt.Errorf("VAE config %s: %w", path, err)
	}
	if n, ok := asInt(raw["latent_channels"]); ok {
		v.LatentChannels = n
	}
	if n, ok := asInt(raw["out_channels"]); ok {
		v.OutChannels = n
	}
	if n, ok := asInt(raw["norm_num_groups"]); ok {
		v.NormNumGroups = n
	}
	if n, ok := asInt(raw["layers_per_block"]); ok {
		v.LayersPerBlock = n
	}
	if f, ok := asFloat(raw["batch_norm_eps"]); ok {
		v.BatchNormEps = float32(f)
	}
	if arr, ok := raw["block_out_channels"].([]any); ok && len(arr) > 0 {
		chs := make([]int, 0, len(arr))
		for _, x := range arr {
			if n, ok := asInt(x); ok {
				chs = append(chs, n)
			}
		}
		if len(chs) > 0 {
			v.BlockOutChannels = chs
		}
	}
	if arr, ok := raw["patch_size"].([]any); ok && len(arr) >= 2 {
		if h, ok := asInt(arr[0]); ok {
			v.PatchH = h
		}
		if w, ok := asInt(arr[1]); ok {
			v.PatchW = w
		}
	}
	return nil
}

func loadConv2d(stPath string, index map[string]hf.TensorInfo, prefix string, pad int) (*conv2d, error) {
	wName := prefix + ".weight"
	ti, ok := index[wName]
	if !ok {
		return nil, fmt.Errorf("missing %s", wName)
	}
	if len(ti.Shape) != 4 {
		return nil, fmt.Errorf("%s: want 4D got %v", wName, ti.Shape)
	}
	outC, inC, kH, kW := ti.Shape[0], ti.Shape[1], ti.Shape[2], ti.Shape[3]
	w, err := hf.LoadF16Vector(stPath, index, wName)
	if err != nil {
		return nil, err
	}
	var bias []float32
	if _, ok := index[prefix+".bias"]; ok {
		bias, err = hf.LoadF16Vector(stPath, index, prefix+".bias")
		if err != nil {
			return nil, err
		}
	}
	return &conv2d{
		OutC: outC, InC: inC, KH: kH, KW: kW, Pad: pad,
		Weight: w, Bias: bias, Name: prefix,
	}, nil
}

func loadGroupNorm(stPath string, index map[string]hf.TensorInfo, prefix string, groups int, eps float32) (*groupNorm, error) {
	w, err := hf.LoadF16Vector(stPath, index, prefix+".weight")
	if err != nil {
		return nil, err
	}
	b, err := hf.LoadF16Vector(stPath, index, prefix+".bias")
	if err != nil {
		return nil, err
	}
	return &groupNorm{
		Groups: groups, Channels: len(w), Eps: eps,
		Weight: w, Bias: b, Name: prefix,
	}, nil
}

func loadResnet(stPath string, index map[string]hf.TensorInfo, prefix string, groups int) (*resnetBlock2D, error) {
	r := &resnetBlock2D{OutputScale: 1, Name: prefix}
	var err error
	if r.Norm1, err = loadGroupNorm(stPath, index, prefix+".norm1", groups, 1e-6); err != nil {
		return nil, err
	}
	if r.Norm2, err = loadGroupNorm(stPath, index, prefix+".norm2", groups, 1e-6); err != nil {
		return nil, err
	}
	if r.Conv1, err = loadConv2d(stPath, index, prefix+".conv1", 1); err != nil {
		return nil, err
	}
	if r.Conv2, err = loadConv2d(stPath, index, prefix+".conv2", 1); err != nil {
		return nil, err
	}
	if _, ok := index[prefix+".conv_shortcut.weight"]; ok {
		if r.ConvShortcut, err = loadConv2d(stPath, index, prefix+".conv_shortcut", 0); err != nil {
			return nil, err
		}
	}
	return r, nil
}

func loadVAEAttention(stPath string, index map[string]hf.TensorInfo, prefix string, channels, groups int) (*vaeAttention, error) {
	gn, err := loadGroupNorm(stPath, index, prefix+".group_norm", groups, 1e-6)
	if err != nil {
		return nil, err
	}
	toQ, err := loadDenseLinear(stPath, index, prefix+".to_q")
	if err != nil {
		return nil, err
	}
	toK, err := loadDenseLinear(stPath, index, prefix+".to_k")
	if err != nil {
		return nil, err
	}
	toV, err := loadDenseLinear(stPath, index, prefix+".to_v")
	if err != nil {
		return nil, err
	}
	toOut, err := loadDenseLinear(stPath, index, prefix+".to_out.0")
	if err != nil {
		return nil, err
	}
	// Diffusers mid-block: attention_head_dim=channels → single head.
	dimHead := channels
	return &vaeAttention{
		Channels:            channels,
		Heads:               1,
		DimHead:             dimHead,
		Scale:               float32(1 / math.Sqrt(float64(dimHead))),
		RescaleOutputFactor: 1,
		GroupNorm:           gn,
		ToQ:                 toQ,
		ToK:                 toK,
		ToV:                 toV,
		ToOut:               toOut,
		Name:                prefix,
	}, nil
}

func loadDenseLinear(stPath string, index map[string]hf.TensorInfo, base string) (*Linear, error) {
	wName := base + ".weight"
	ti, ok := index[wName]
	if !ok {
		return nil, fmt.Errorf("missing %s", wName)
	}
	if len(ti.Shape) != 2 {
		return nil, fmt.Errorf("%s: expected 2D got %v", wName, ti.Shape)
	}
	w, err := hf.LoadF16Vector(stPath, index, wName)
	if err != nil {
		return nil, err
	}
	var bias []float32
	if _, ok := index[base+".bias"]; ok {
		bias, err = hf.LoadF16Vector(stPath, index, base+".bias")
		if err != nil {
			return nil, err
		}
	}
	return NewDenseLinear(ti.Shape[0], ti.Shape[1], w, bias, base)
}

func reverseInts(xs []int) []int {
	out := make([]int, len(xs))
	for i := range xs {
		out[i] = xs[len(xs)-1-i]
	}
	return out
}
