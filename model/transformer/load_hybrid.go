package transformer

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/fusedgpu"
	"github.com/openfluke/welvet/layers/gdn"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/model/entity"
	"github.com/openfluke/welvet/quant"
)

// loadHybridEntity loads Qwen3.5 / Bonsai text tower from ENTITY.
func loadHybridEntity(ef *entity.File, m *Model, spec *entity.TransformerSpec) error {
	d := spec.Dims
	packFmt := quant.FormatBinaryPacked
	m.PackFormat = packFmt
	m.FusedPack = true
	m.Architecture = spec.Architecture
	m.LayerTypes = append([]string(nil), d.LayerTypes...)
	m.AttnOutputGate = d.AttnOutputGate
	m.PartialRotary = d.PartialRotaryFactor
	if m.PartialRotary <= 0 {
		m.PartialRotary = 1
	}
	m.MaxSeqLen = fusedgpu.ClampHostMaxSeq(m.MaxSeqLen)

	embBlob, err := ef.LoadQuantBlob("transformer.embeddings.packed")
	if err != nil {
		return fmt.Errorf("embed packed: %w", err)
	}
	m.embedPacked = embBlob

	if err := loadLMHeadPacked(ef, m); err != nil {
		return err
	}

	eps := d.RMSNormEps
	if eps <= 0 {
		eps = 1e-6
	}
	if spec.HasFinalNorm {
		fn, err := ef.LoadBlob("transformer.final_norm")
		if err != nil {
			return err
		}
		m.FinalNorm, err = rmsnorm.NewConfigured(rmsnorm.Config{Dim: spec.HiddenSize, Eps: eps}, core.DTypeFloat32, quant.FormatNone, fn)
		if err != nil {
			return err
		}
	}

	hidden := spec.HiddenSize
	inter := d.IntermediateSize
	headDim := d.HeadDim
	qRows := d.NumHeads * headDim
	if d.AttnOutputGate {
		qRows *= 2
	}
	kvRows := d.NumKVHeads * headDim

	linCfg := gdn.Config{
		HiddenSize:    hidden,
		NumKeyHeads:   d.LinearNumKeyHeads,
		NumValueHeads: d.LinearNumValueHeads,
		KeyHeadDim:    d.LinearKeyHeadDim,
		ValueHeadDim:  d.LinearValueHeadDim,
		ConvKernel:    d.LinearConvKernel,
		Eps:           eps,
	}

	for i := 0; i < d.NumLayers; i++ {
		prefix := fmt.Sprintf("blocks.%d", i)
		lt := "full_attention"
		if i < len(d.LayerTypes) {
			lt = d.LayerTypes[i]
		}
		an, err := ef.LoadBlob(prefix + ".attn_norm")
		if err != nil {
			return err
		}
		attnNorm, err := rmsnorm.NewConfigured(rmsnorm.Config{Dim: hidden, Eps: eps}, core.DTypeFloat32, quant.FormatNone, an)
		if err != nil {
			return err
		}
		fn, err := ef.LoadBlob(prefix + ".ffn_norm")
		if err != nil {
			return err
		}
		ffnNorm, err := rmsnorm.NewConfigured(rmsnorm.Config{Dim: hidden, Eps: eps}, core.DTypeFloat32, quant.FormatNone, fn)
		if err != nil {
			return err
		}
		gateStore, err := loadWeightStore(ef, prefix+".gate", inter, hidden, packFmt)
		if err != nil {
			return err
		}
		upStore, err := loadWeightStore(ef, prefix+".up", inter, hidden, packFmt)
		if err != nil {
			return err
		}
		downStore, err := loadWeightStore(ef, prefix+".down", hidden, inter, packFmt)
		if err != nil {
			return err
		}
		ffn := &swiglu.Layer{
			Core: core.Layer{
				Type: core.LayerSwiGLU, DType: core.DTypeFloat32,
				InputHeight: hidden, OutputHeight: hidden, TileSize: 32, MultiCore: true,
			},
			Cfg:  swiglu.Config{InputDim: hidden, IntermediateDim: inter},
			Exec: core.ExecConfig{Backend: core.BackendCPUTiled, MultiCore: true, TileSize: 32},
			Gate: denseFromStore(hidden, inter, core.ActivationLinear, gateStore),
			Up:   denseFromStore(hidden, inter, core.ActivationLinear, upStore),
			Down: denseFromStore(inter, hidden, core.ActivationLinear, downStore),
		}

		blk := Block{AttnNorm: attnNorm, FFNNorm: ffnNorm, FFN: ffn, LayerType: lt}

		switch lt {
		case "full_attention":
			qStore, err := loadWeightStore(ef, prefix+".q", qRows, hidden, packFmt)
			if err != nil {
				return err
			}
			kStore, err := loadWeightStore(ef, prefix+".k", kvRows, hidden, packFmt)
			if err != nil {
				return err
			}
			vStore, err := loadWeightStore(ef, prefix+".v", kvRows, hidden, packFmt)
			if err != nil {
				return err
			}
			oStore, err := loadWeightStore(ef, prefix+".o", hidden, d.NumHeads*headDim, packFmt)
			if err != nil {
				return err
			}
			qn, err := ef.LoadBlob(prefix + ".q_norm")
			if err != nil {
				return err
			}
			kn, err := ef.LoadBlob(prefix + ".k_norm")
			if err != nil {
				return err
			}
			blk.Q = denseFromStore(hidden, qRows, core.ActivationLinear, qStore)
			blk.K = denseFromStore(hidden, kvRows, core.ActivationLinear, kStore)
			blk.V = denseFromStore(hidden, kvRows, core.ActivationLinear, vStore)
			blk.O = denseFromStore(d.NumHeads*headDim, hidden, core.ActivationLinear, oStore)
			blk.QNorm = qn
			blk.KNorm = kn
			blk.NumHeads = d.NumHeads
			blk.NumKVHeads = d.NumKVHeads
			blk.HeadDim = headDim
			blk.RoPETheta = d.RoPEFreqBase
			blk.PartialRotary = m.PartialRotary
			blk.OutputGate = d.AttnOutputGate
			blk.KVCacheK = nil
			blk.KVCacheV = nil
		case "linear_attention":
			loadB := func(name string) (*quant.Blob, error) {
				return ef.LoadQuantBlob(prefix + "." + name)
			}
			inQKV, err := loadB("gdn_qkv")
			if err != nil {
				return err
			}
			inZ, err := loadB("gdn_z")
			if err != nil {
				return err
			}
			inB, err := loadB("gdn_b")
			if err != nil {
				return err
			}
			inA, err := loadB("gdn_a")
			if err != nil {
				return err
			}
			outP, err := loadB("gdn_out")
			if err != nil {
				return err
			}
			conv, err := ef.LoadBlob(prefix + ".gdn_conv")
			if err != nil {
				return err
			}
			aLog, err := ef.LoadBlob(prefix + ".gdn_A_log")
			if err != nil {
				return err
			}
			dt, err := ef.LoadBlob(prefix + ".gdn_dt_bias")
			if err != nil {
				return err
			}
			gn, err := ef.LoadBlob(prefix + ".gdn_norm")
			if err != nil {
				return err
			}
			blk.GDN = &gdn.Layer{
				Cfg:        linCfg,
				InQKV:      inQKV,
				InZ:        inZ,
				InB:        inB,
				InA:        inA,
				Out:        outP,
				ConvWeight: conv,
				ALog:       aLog,
				DtBias:     dt,
				NormGamma:  gn,
			}
		default:
			return fmt.Errorf("layer %d: unknown type %q", i, lt)
		}
		m.Blocks[i] = blk
	}
	unbakeHybridNormsIfNeeded(m)
	return nil
}

// unbakeHybridNormsIfNeeded undoes a mistaken extra (1+w) bake on entities that
// were packed twice. MLX Bonsai already stores multiplicative γ (not HF OffsetRMS).
//
// Dense Bonsai final γ often sits near ~2.0–2.8 naturally; early layer norms can
// be near 0.02. Thresholding on final mean alone false-triggers and destroys the
// model (subtracting 1 from tiny layer norms). Only unbake when layer-0 attn norm
// also looks double-baked (mean ≳ 1.5), which marks the bad pack path.
func unbakeHybridNormsIfNeeded(m *Model) {
	if m == nil || m.FinalNorm == nil || m.FinalNorm.Gamma == nil {
		return
	}
	g, ok := m.FinalNorm.Gamma.MasterF32()
	if !ok || len(g) == 0 {
		return
	}
	finalMean := meanF32(g)
	if finalMean < 2.4 {
		return
	}
	layer0Mean := 0.0
	if len(m.Blocks) > 0 && m.Blocks[0].AttnNorm != nil && m.Blocks[0].AttnNorm.Gamma != nil {
		if w, ok := m.Blocks[0].AttnNorm.Gamma.MasterF32(); ok && len(w) > 0 {
			layer0Mean = meanF32(w)
		}
	}
	// Healthy dense Bonsai: final ~2.5+, layer0 attn ~0.02. Double-baked: both high.
	if layer0Mean < 1.5 {
		return
	}
	fmt.Printf("  undoing double-baked RMSNorm (final γ mean=%.2f → %.2f, layer0 attn=%.2f)\n",
		finalMean, finalMean-1, layer0Mean)
	sub1 := func(s []float32) {
		for i := range s {
			s[i]--
		}
	}
	sub1Store := func(l *rmsnorm.Layer) {
		if l == nil || l.Gamma == nil {
			return
		}
		if w, ok := l.Gamma.MasterF32(); ok {
			sub1(w)
		}
	}
	sub1Store(m.FinalNorm)
	for i := range m.Blocks {
		b := &m.Blocks[i]
		sub1Store(b.AttnNorm)
		sub1Store(b.FFNNorm)
		sub1(b.QNorm)
		sub1(b.KNorm)
		if b.GDN != nil {
			sub1(b.GDN.NormGamma)
		}
	}
}

func meanF32(s []float32) float64 {
	if len(s) == 0 {
		return 0
	}
	var sum float64
	for _, v := range s {
		sum += float64(v)
	}
	return sum / float64(len(s))
}

func loadLMHeadPacked(ef *entity.File, m *Model) error {
	b, err := ef.LoadQuantBlob("transformer.lm_head.packed")
	if err != nil {
		// Tied embeddings: reuse packed embed table (Bonsai 4B / 1.7B).
		if m.embedPacked != nil {
			m.lmHeadPacked = m.embedPacked
			m.LMHeadTied = true
			return nil
		}
		ws, err2 := loadWeightStore(ef, "transformer.lm_head", m.VocabSize, m.HiddenSize, quant.FormatBinaryPacked)
		if err2 != nil {
			return err
		}
		m.lmHead = ws
		return nil
	}
	m.lmHeadPacked = b
	m.LMHeadTied = false
	return nil
}

// gatherEmbedPacked writes token embedding rows into dst [nTok*hidden].
func gatherEmbedPacked(b *quant.Blob, ids []uint32, dst []float32) error {
	h := b.Cols
	for t, id := range ids {
		if int(id) >= b.Rows {
			return fmt.Errorf("embed: token %d OOB (vocab %d)", id, b.Rows)
		}
		if err := quant.DecodeRow(b, int(id), dst[t*h:(t+1)*h]); err != nil {
			return err
		}
	}
	return nil
}
