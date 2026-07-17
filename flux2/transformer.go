package flux2

import (
	"fmt"
	"math"
)

// Model is Flux2Transformer2DModel weights + forward.
type Model struct {
	Cfg Config

	// UseGPU is set by SyncGPU when BinaryG128 weights are resident in VRAM.
	UseGPU bool

	XEmbedder       *Linear
	ContextEmbedder *Linear

	// TimestepEmbedding: linear_1 [inner, 256], linear_2 [inner, inner]
	TimeLinear1 *Linear
	TimeLinear2 *Linear

	DoubleModImg *Linear // SiLU(temb) → dim*6
	DoubleModTxt *Linear
	SingleMod    *Linear // → dim*3

	DoubleBlocks []DoubleStreamBlock
	SingleBlocks []SingleStreamBlock

	// AdaLayerNormContinuous: linear [2*dim, dim] after SiLU(temb)
	NormOutLinear *Linear
	ProjOut       *Linear
}

// Forward runs the MMDiT.
// hiddenStates: [imgSeq * inChannels], encoderHiddenStates: [txtSeq * jointDim]
// imgIds: [imgSeq * 4], txtIds: [txtSeq * 4] (axes_dims_rope)
// timestep is the scheduler timestep value (already in schedule units; multiplied by 1000 internally like diffusers).
func (m *Model) Forward(
	hiddenStates, encoderHiddenStates []float32,
	timestep float32,
	imgIds, txtIds []float32,
	imgSeq, txtSeq int,
) ([]float32, error) {
	if m == nil {
		return nil, fmt.Errorf("Model.Forward: nil model")
	}
	cfg := m.Cfg
	dim := cfg.InnerDim()
	if len(hiddenStates) < imgSeq*cfg.InChannels {
		return nil, fmt.Errorf("hidden_states short")
	}
	if len(encoderHiddenStates) < txtSeq*cfg.JointAttentionDim {
		return nil, fmt.Errorf("encoder_hidden_states short")
	}

	// 1. timestep embedding (*1000 like diffusers)
	temb, err := m.embedTimestep(float64(timestep) * 1000)
	if err != nil {
		return nil, err
	}

	modImg := make([]float32, dim*6)
	modTxt := make([]float32, dim*6)
	modSingle := make([]float32, dim*3)
	siluTemb := SiLUCopy(temb)
	if err := m.DoubleModImg.MatVec(siluTemb, modImg); err != nil {
		return nil, err
	}
	if err := m.DoubleModTxt.MatVec(siluTemb, modTxt); err != nil {
		return nil, err
	}
	if err := m.SingleMod.MatVec(siluTemb, modSingle); err != nil {
		return nil, err
	}

	// 2. input projections
	hidden := make([]float32, imgSeq*dim)
	encoder := make([]float32, txtSeq*dim)
	if err := m.XEmbedder.MatMulSeq(hiddenStates, hidden, imgSeq); err != nil {
		return nil, err
	}
	if err := m.ContextEmbedder.MatMulSeq(encoderHiddenStates, encoder, txtSeq); err != nil {
		return nil, err
	}

	// 3. RoPE
	imgRope := PosEmbedND(imgIds, imgSeq, cfg.AxesDimsRope, cfg.RopeTheta)
	txtRope := PosEmbedND(txtIds, txtSeq, cfg.AxesDimsRope, cfg.RopeTheta)
	rope := ConcatRotary(txtRope, imgRope)

	// 4. double stream
	for i := range m.DoubleBlocks {
		if err := m.DoubleBlocks[i].Forward(hidden, encoder, imgSeq, txtSeq, modImg, modTxt, rope); err != nil {
			return nil, fmt.Errorf("double block %d: %w", i, err)
		}
	}

	// concat txt||img for single stream
	total := txtSeq + imgSeq
	joint := make([]float32, total*dim)
	copy(joint, encoder)
	copy(joint[txtSeq*dim:], hidden)

	// 5. single stream
	for i := range m.SingleBlocks {
		if err := m.SingleBlocks[i].Forward(joint, total, modSingle, rope); err != nil {
			return nil, fmt.Errorf("single block %d: %w", i, err)
		}
	}

	// drop text tokens
	hidden = joint[txtSeq*dim : total*dim]

	// 6. AdaLayerNormContinuous + proj_out
	scaleShift := make([]float32, 2*dim)
	if err := m.NormOutLinear.MatVec(SiLUCopy(temb), scaleShift); err != nil {
		return nil, err
	}
	scale := scaleShift[:dim]
	shift := scaleShift[dim:]
	AdaLayerNormContinuous(hidden, scale, shift, imgSeq, dim, cfg.Eps)

	outCh := cfg.OutChannels
	if outCh == 0 {
		outCh = cfg.InChannels
	}
	out := make([]float32, imgSeq*outCh)
	if err := m.ProjOut.MatMulSeq(hidden, out, imgSeq); err != nil {
		return nil, err
	}
	return out, nil
}

func (m *Model) embedTimestep(t float64) ([]float32, error) {
	channels := m.Cfg.TimestepGuidanceChannels
	if channels == 0 {
		channels = 256
	}
	proj := getTimestepEmbedding(t, channels, true, 0, 1)
	h := make([]float32, m.Cfg.InnerDim())
	if err := m.TimeLinear1.MatVec(proj, h); err != nil {
		return nil, err
	}
	SiLU(h)
	out := make([]float32, m.Cfg.InnerDim())
	if err := m.TimeLinear2.MatVec(h, out); err != nil {
		return nil, err
	}
	return out, nil
}

// getTimestepEmbedding ports diffusers get_timestep_embedding for a scalar timestep.
// With flipSinToCos=true (Flux2 Timesteps), output is [cos..., sin...] after the half-swap.
func getTimestepEmbedding(timestep float64, embeddingDim int, flipSinToCos bool, downscaleFreqShift, scale float64) []float32 {
	half := embeddingDim / 2
	emb := make([]float32, embeddingDim)
	for i := 0; i < half; i++ {
		exponent := -math.Log(10000) * float64(i) / (float64(half) - downscaleFreqShift)
		freq := math.Exp(exponent)
		angle := scale * timestep * freq
		emb[i] = float32(math.Sin(angle))
		emb[half+i] = float32(math.Cos(angle))
	}
	if flipSinToCos {
		out := make([]float32, embeddingDim)
		copy(out[:half], emb[half:])
		copy(out[half:], emb[:half])
		return out
	}
	return emb
}
