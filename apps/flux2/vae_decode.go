package flux2

import (
	"fmt"
	"time"
)

// decodeNCHW runs post_quant_conv + Decoder on VAE latents [C,H,W].
// Matches AutoencoderKLFlux2._decode (no tiling/slicing).
func (v *AutoencoderKLFlux2) decodeNCHW(z nchw) (nchw, error) {
	t0 := time.Now()
	log := func(name string, t nchw) {
		fmt.Printf("    VAE %-16s %dx%d×%d  (+%v)\n", name, t.H, t.W, t.C, time.Since(t0).Round(time.Millisecond))
		t0 = time.Now()
	}

	if v.PostQuant != nil {
		var err error
		z, err = v.PostQuant.forward(z)
		if err != nil {
			return nchw{}, fmt.Errorf("post_quant_conv: %w", err)
		}
		log("post_quant", z)
	}
	h, err := v.ConvIn.forward(z)
	if err != nil {
		return nchw{}, fmt.Errorf("decoder.conv_in: %w", err)
	}
	log("conv_in", h)
	h, err = v.Mid.forward(h)
	if err != nil {
		return nchw{}, fmt.Errorf("decoder.mid_block: %w", err)
	}
	log("mid_block", h)
	for i, up := range v.UpBlocks {
		h, err = up.forward(h)
		if err != nil {
			return nchw{}, fmt.Errorf("decoder.up_blocks.%d: %w", i, err)
		}
		log(fmt.Sprintf("up_blocks.%d", i), h)
	}
	h, err = v.NormOut.forward(h)
	if err != nil {
		return nchw{}, fmt.Errorf("decoder.conv_norm_out: %w", err)
	}
	siluInPlace(h.Data)
	h, err = v.ConvOut.forward(h)
	if err != nil {
		return nchw{}, fmt.Errorf("decoder.conv_out: %w", err)
	}
	log("conv_out", h)
	return h, nil
}
