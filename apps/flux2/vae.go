package flux2

import (
	"fmt"
	"math"
)

// AutoencoderKLFlux2 is the Flux2 VAE decoder (diffusers AutoencoderKLFlux2).
// Encode path is not ported — only post_quant_conv + decoder (+ BN stats for Klein denorm).
//
// Flux2 Klein denormalization uses BatchNorm running stats on *patchified* (C=128) latents,
// not the classic AutoencoderKL scaling_factor. See DenormalizePacked / pipeline.Generate.
type AutoencoderKLFlux2 struct {
	LatentChannels   int
	OutChannels      int
	BlockOutChannels []int
	NormNumGroups    int
	BatchNormEps     float32
	PatchH, PatchW   int
	LayersPerBlock   int
	Loaded           bool

	// BN running stats over patch_size² * latent_channels (=128 for 2×2×32).
	BNMean []float32
	BNVar  []float32

	PostQuant *conv2d
	ConvIn    *conv2d
	Mid       *vaeMidBlock
	UpBlocks  []*vaeUpBlock
	NormOut   *groupNorm
	ConvOut   *conv2d

	// Legacy fields kept for API compatibility; Flux2 Klein does not use them for decode.
	ScalingFactor float32
	ShiftFactor   float32

	// gpu is the fused WebGPU decoder (set by SyncGPU).
	gpu *vaeGPU
}

// NewVAEStub returns an unloaded VAE placeholder matching Flux2 Klein defaults.
func NewVAEStub() *AutoencoderKLFlux2 {
	return &AutoencoderKLFlux2{
		LatentChannels:   32,
		OutChannels:      3,
		BlockOutChannels: []int{128, 256, 512, 512},
		NormNumGroups:    32,
		BatchNormEps:     1e-4,
		PatchH:           2,
		PatchW:           2,
		LayersPerBlock:   2,
		ScalingFactor:    1, // unused by Flux2 BN denorm path
		ShiftFactor:      0,
		Loaded:           false,
	}
}

// ErrVAENotImplemented is returned when Decode is called without loaded weights.
var ErrVAENotImplemented = fmt.Errorf("VAE decode: weights not loaded (call LoadVAEFromDir)")

// Decode maps VAE-native latents [latent_channels × latH × latW] (CHW flat) → RGB
// float32 HWC [outH×outW×3] in [0,1], where outH=latH*8 and outW=latW*8 for the
// default 4-level decoder (vae_scale_factor = 2^(len(block_out_channels)-1) = 8).
//
// Caller must unpack transformer latents, BN-denormalize, and unpatchify first
// (see UnpackLatentsCHW, DenormalizePacked, UnpatchifyLatents).
func (v *AutoencoderKLFlux2) Decode(latents []float32, latH, latW int) ([]float32, error) {
	if v == nil || !v.Loaded {
		return nil, ErrVAENotImplemented
	}
	if latH < 1 || latW < 1 {
		return nil, fmt.Errorf("VAE.Decode: bad spatial %dx%d", latH, latW)
	}
	c := v.LatentChannels
	need := c * latH * latW
	if len(latents) < need {
		return nil, fmt.Errorf("VAE.Decode: latents short %d need %d", len(latents), need)
	}
	if v.gpu == nil || !v.gpu.ready {
		return nil, fmt.Errorf("VAE.Decode: GPU not synced (call SyncGPU) — no CPU fallback")
	}
	return v.gpu.decode(v, latents[:need], latH, latW)
}

// ScaleLatents is the classic AutoencoderKL denorm (latents/scaling + shift).
// Flux2 Klein pipelines use DenormalizePacked (BN) instead — kept for compatibility.
func (v *AutoencoderKLFlux2) ScaleLatents(latents []float32) {
	if v == nil || v.ScalingFactor == 0 {
		return
	}
	inv := 1.0 / v.ScalingFactor
	for i := range latents {
		latents[i] = latents[i]*inv + v.ShiftFactor
	}
}

// DenormalizePacked applies Flux2KleinPipeline BN denorm on patchified latents:
//   z = z * sqrt(running_var + eps) + running_mean
// expecting CHW flat with C = patch_h*patch_w*latent_channels (128).
func (v *AutoencoderKLFlux2) DenormalizePacked(latents []float32, packedH, packedW int) error {
	if v == nil {
		return fmt.Errorf("DenormalizePacked: nil VAE")
	}
	c := v.PatchH * v.PatchW * v.LatentChannels
	if len(v.BNMean) < c || len(v.BNVar) < c {
		return fmt.Errorf("DenormalizePacked: missing BN stats (need LoadVAEFromDir)")
	}
	need := c * packedH * packedW
	if len(latents) < need {
		return fmt.Errorf("DenormalizePacked: short %d need %d", len(latents), need)
	}
	hw := packedH * packedW
	eps := v.BatchNormEps
	for ch := 0; ch < c; ch++ {
		std := float32(math.Sqrt(float64(v.BNVar[ch] + eps)))
		mean := v.BNMean[ch]
		base := ch * hw
		for i := 0; i < hw; i++ {
			latents[base+i] = latents[base+i]*std + mean
		}
	}
	return nil
}

// UnpackLatentsCHW converts packed transformer latents [seq×channels] (row-major
// y*W+x) into CHW flat [C×H×W].
func UnpackLatentsCHW(packed []float32, channels, h, w int) ([]float32, error) {
	seq := h * w
	need := seq * channels
	if len(packed) < need {
		return nil, fmt.Errorf("UnpackLatentsCHW: short %d need %d", len(packed), need)
	}
	out := make([]float32, need)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			s := y*w + x
			for c := 0; c < channels; c++ {
				out[c*seq+s] = packed[s*channels+c]
			}
		}
	}
	return out, nil
}

// UnpatchifyLatents undoes Flux2 2×2 patchify:
// [C*4, H, W] → [C, H*2, W*2] (Diffusers Flux2Pipeline._unpatchify_latents).
func UnpatchifyLatents(latents []float32, packedChannels, h, w int) (out []float32, outH, outW, outC int, err error) {
	if packedChannels%(2*2) != 0 {
		return nil, 0, 0, 0, fmt.Errorf("UnpatchifyLatents: channels %d not divisible by 4", packedChannels)
	}
	outC = packedChannels / 4
	need := packedChannels * h * w
	if len(latents) < need {
		return nil, 0, 0, 0, fmt.Errorf("UnpatchifyLatents: short %d need %d", len(latents), need)
	}
	outH, outW = h*2, w*2
	out = make([]float32, outC*outH*outW)
	inHW := h * w
	outHW := outH * outW
	for c := 0; c < outC; c++ {
		for ph := 0; ph < 2; ph++ {
			for pw := 0; pw < 2; pw++ {
				inC := c*4 + ph*2 + pw
				src := latents[inC*inHW : (inC+1)*inHW]
				for y := 0; y < h; y++ {
					for x := 0; x < w; x++ {
						oy := y*2 + ph
						ox := x*2 + pw
						out[c*outHW+oy*outW+ox] = src[y*w+x]
					}
				}
			}
		}
	}
	return out, outH, outW, outC, nil
}

// FloatRGBToUint8 converts HWC float RGB in [0,1] to uint8.
func FloatRGBToUint8(rgb []float32) []uint8 {
	out := make([]uint8, len(rgb))
	for i, v := range rgb {
		if v < 0 {
			v = 0
		} else if v > 1 {
			v = 1
		}
		out[i] = uint8(v*255 + 0.5)
	}
	return out
}
