package flux2

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
)

// Pipeline wires scheduler + transformer + AutoencoderKLFlux2 for Klein-style generation.
type Pipeline struct {
	Transformer *Model
	Scheduler   *FlowMatchEulerDiscreteScheduler
	VAE         *AutoencoderKLFlux2
	Cfg         Config
}

// NewPipeline builds a pipeline from a loaded transformer (VAE stub until LoadVAEFromDir).
func NewPipeline(m *Model) *Pipeline {
	cfg := DefaultConfig()
	if m != nil {
		cfg = m.Cfg
	}
	return &Pipeline{
		Transformer: m,
		Scheduler:   NewFlowMatchEulerDiscreteScheduler(3.0, true),
		VAE:         NewVAEStub(),
		Cfg:         cfg,
	}
}

// LoadVAE loads AutoencoderKLFlux2 weights from snapshotDir/vae/.
func (p *Pipeline) LoadVAE(snapshotDir string) error {
	if p == nil {
		return fmt.Errorf("Pipeline.LoadVAE: nil")
	}
	v, err := LoadVAEFromDir(snapshotDir)
	if err != nil {
		return err
	}
	p.VAE = v
	return nil
}

// Generate runs the flow-match denoising loop then VAE-decodes to PNG.
// promptEmbeds is [txtSeq * jointAttentionDim] (caller supplies text encoder output).
// When VAE weights are loaded, returns a real RGB PNG and nil error.
// When VAE is unloaded, returns a latent-visualized placeholder PNG and ErrVAENotImplemented.
func (p *Pipeline) Generate(
	promptEmbeds []float32,
	txtSeq int,
	height, width, steps int,
	seed int64,
) ([]byte, error) {
	if p == nil || p.Transformer == nil {
		return nil, fmt.Errorf("Pipeline.Generate: nil transformer")
	}
	if steps <= 0 {
		steps = 4
	}
	if height <= 0 {
		height = 512
	}
	if width <= 0 {
		width = 512
	}
	cfg := p.Cfg
	// Flux2 packs 2x2 latent patches; VAE scale 8 → packed H/W = pixel/16
	latH := height / 16
	latW := width / 16
	if latH < 1 {
		latH = 1
	}
	if latW < 1 {
		latW = 1
	}
	imgSeq := latH * latW
	inCh := cfg.InChannels

	rng := rand.New(rand.NewSource(seed))
	latents := make([]float32, imgSeq*inCh)
	for i := range latents {
		latents[i] = float32(rng.NormFloat64())
	}

	imgIds := makeImageIds(latH, latW)
	txtIds := makeTextIds(txtSeq)

	mu := ComputeEmpiricalMu(imgSeq, steps)
	if err := p.Scheduler.SetTimesteps(steps, mu); err != nil {
		return nil, err
	}
	p.Scheduler.ResetStepIndex()

	for i, t := range p.Scheduler.Timesteps {
		fmt.Printf("  denoise step %d/%d (t=%.4g)…\n", i+1, len(p.Scheduler.Timesteps), t)
		noisePred, err := p.Transformer.Forward(
			latents, promptEmbeds, float32(t), imgIds, txtIds, imgSeq, txtSeq,
		)
		if err != nil {
			return nil, fmt.Errorf("transformer forward: %w", err)
		}
		latents, err = p.Scheduler.Step(noisePred, latents, t)
		if err != nil {
			return nil, fmt.Errorf("scheduler step: %w", err)
		}
	}

	if p.VAE == nil || p.VAE.gpu == nil || !p.VAE.gpu.ready {
		return nil, fmt.Errorf("Pipeline.Generate: VAE GPU not synced (call VAE.SyncGPU)")
	}
	fmt.Println("  VAE decode (GPU fuse)…")
	png, err := p.decodeLatentsPNG(latents, inCh, latH, latW, height, width)
	return png, err
}

// decodeLatentsPNG unpacks → BN-denorm → unpatchify → VAE.Decode → PNG.
func (p *Pipeline) decodeLatentsPNG(packed []float32, channels, latH, latW, height, width int) ([]byte, error) {
	if p.VAE == nil || !p.VAE.Loaded {
		return nil, ErrVAENotImplemented
	}
	chw, err := UnpackLatentsCHW(packed, channels, latH, latW)
	if err != nil {
		return nil, err
	}
	if err := p.VAE.DenormalizePacked(chw, latH, latW); err != nil {
		return nil, err
	}
	vaeZ, vh, vw, _, err := UnpatchifyLatents(chw, channels, latH, latW)
	if err != nil {
		return nil, err
	}
	rgbF, err := p.VAE.Decode(vaeZ, vh, vw)
	if err != nil {
		return nil, err
	}
	outH, outW := vh*8, vw*8
	if len(rgbF) < outH*outW*3 {
		outH, outW = height, width
	}
	rgb := FloatRGBToUint8(rgbF)
	var buf bytes.Buffer
	if err := EncodePNG(&buf, rgb, outH, outW); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func makeImageIds(h, w int) []float32 {
	// ids: [seq, 4] — typically (t, h, w, …) style; Flux2 uses 4 axes.
	ids := make([]float32, h*w*4)
	i := 0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			ids[i*4+0] = 0
			ids[i*4+1] = float32(y)
			ids[i*4+2] = float32(x)
			ids[i*4+3] = 0
			i++
		}
	}
	return ids
}

func makeTextIds(seq int) []float32 {
	ids := make([]float32, seq*4)
	for i := 0; i < seq; i++ {
		ids[i*4+0] = 0
		ids[i*4+1] = float32(i)
		ids[i*4+2] = 0
		ids[i*4+3] = 0
	}
	return ids
}

func clamp255(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	if math.IsNaN(float64(v)) {
		return 0
	}
	return v
}
