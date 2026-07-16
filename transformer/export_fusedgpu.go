package transformer

import (
	"fmt"

	"github.com/openfluke/welvet/dense"
	"github.com/openfluke/welvet/fusedgpu"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/rmsnorm"
	"github.com/openfluke/welvet/weights"
)

// ExportFusedGPUSpec builds the weight bundle for the full-stack Q4 GPU decoder.
func (m *Model) ExportFusedGPUSpec() (*fusedgpu.Spec, error) {
	if m == nil {
		return nil, fmt.Errorf("transformer: nil model")
	}
	if !m.FusedGPUReady() {
		return nil, fmt.Errorf("transformer: need baked Q4_0 entity")
	}
	if len(m.Blocks) == 0 {
		return nil, fmt.Errorf("transformer: no blocks")
	}
	cfg := m.Blocks[0].Attn.Cfg
	headDim := cfg.HeadDim
	if headDim <= 0 {
		headDim = m.HiddenSize / cfg.NumHeads
	}
	eps := float32(cfg.QKNormEps)
	if m.Blocks[0].AttnNorm != nil && m.Blocks[0].AttnNorm.Cfg.Eps > 0 {
		eps = float32(m.Blocks[0].AttnNorm.Cfg.Eps)
	}
	if eps <= 0 {
		eps = 1e-5
	}
	rope := float32(cfg.RoPETheta)
	if rope <= 0 {
		rope = 10000
	}
	maxSeq := cfg.MaxSeqLen
	if maxSeq <= 0 {
		maxSeq = m.MaxSeqLen
	}
	if maxSeq <= 0 {
		maxSeq = 512
	}

	emb, err := weights.MatrixF32(m.Embed.Weights)
	if err != nil {
		return nil, fmt.Errorf("embed: %w", err)
	}

	spec := &fusedgpu.Spec{
		Hidden:       m.HiddenSize,
		Vocab:        m.VocabSize,
		Layers:       len(m.Blocks),
		Heads:        cfg.NumHeads,
		KVHeads:      cfg.NumKVHeads,
		HeadDim:      headDim,
		QDim:         cfg.QDim(),
		KVDim:        cfg.KVDim(),
		Intermediate: m.Blocks[0].FFN.Cfg.IntermediateDim,
		Eps:          eps,
		RopeTheta:    rope,
		MaxSeq:       maxSeq,
		Embed:        emb,
	}
	if m.HasFinalNorm && m.FinalNorm != nil {
		spec.FinalNorm, err = rmsGammaF32(m.FinalNorm)
		if err != nil {
			return nil, fmt.Errorf("final_norm: %w", err)
		}
	} else {
		spec.FinalNorm = onesF32(m.HiddenSize)
	}
	if err := loadLMHeadSpec(m, spec); err != nil {
		return nil, err
	}

	spec.Blocks = make([]fusedgpu.BlockSpec, len(m.Blocks))
	for i, b := range m.Blocks {
		blk := &spec.Blocks[i]
		if b.AttnNorm != nil {
			blk.AttnNorm, err = rmsGammaF32(b.AttnNorm)
			if err != nil {
				return nil, fmt.Errorf("block %d attn_norm: %w", i, err)
			}
		}
		if b.FFNNorm != nil {
			blk.MLPNorm, err = rmsGammaF32(b.FFNNorm)
			if err != nil {
				return nil, fmt.Errorf("block %d ffn_norm: %w", i, err)
			}
		}
		blk.Q, err = q4SpecFromDense(b.Attn.Q)
		if err != nil {
			return nil, fmt.Errorf("block %d Q: %w", i, err)
		}
		blk.K, err = q4SpecFromDense(b.Attn.K)
		if err != nil {
			return nil, fmt.Errorf("block %d K: %w", i, err)
		}
		blk.V, err = q4SpecFromDense(b.Attn.V)
		if err != nil {
			return nil, fmt.Errorf("block %d V: %w", i, err)
		}
		blk.O, err = q4SpecFromDense(b.Attn.O)
		if err != nil {
			return nil, fmt.Errorf("block %d O: %w", i, err)
		}
		blk.Gate, err = q4SpecFromDense(b.FFN.Gate)
		if err != nil {
			return nil, fmt.Errorf("block %d gate: %w", i, err)
		}
		blk.Up, err = q4SpecFromDense(b.FFN.Up)
		if err != nil {
			return nil, fmt.Errorf("block %d up: %w", i, err)
		}
		blk.Down, err = q4SpecFromDense(b.FFN.Down)
		if err != nil {
			return nil, fmt.Errorf("block %d down: %w", i, err)
		}
	}
	return spec, nil
}

func loadLMHeadSpec(m *Model, spec *fusedgpu.Spec) error {
	if blob := m.LMHeadPackedBlob(); blob != nil {
		scales, packed, ok := quant.Q4SIMD(blob)
		if !ok {
			return fmt.Errorf("lm_head packed Q4 missing SIMD cache")
		}
		spec.LMScales = scales
		spec.LMPacked = packed
		return nil
	}
	if head := m.UntiedLMHead(); head != nil && head.Format == quant.FormatQ4_0 && head.Packed != nil {
		scales, packed, ok := quant.Q4SIMD(head.Packed)
		if !ok {
			return fmt.Errorf("lm_head Q4 missing SIMD cache")
		}
		spec.LMScales = scales
		spec.LMPacked = packed
		return nil
	}
	need := m.VocabSize * m.HiddenSize
	if len(spec.Embed) < need {
		return fmt.Errorf("lm_head embed short: %d need %d", len(spec.Embed), need)
	}
	blob, err := quant.Pack(quant.FormatQ4_0, spec.Embed[:need], m.VocabSize, m.HiddenSize)
	if err != nil {
		return err
	}
	quant.EnsureQ4SIMDCache(blob)
	scales, packed, ok := quant.Q4SIMD(blob)
	if !ok {
		return fmt.Errorf("lm_head repack failed")
	}
	spec.LMScales = scales
	spec.LMPacked = packed
	return nil
}

func q4SpecFromDense(l *dense.Layer) (fusedgpu.Q4Spec, error) {
	if l == nil || l.Weights == nil {
		return fusedgpu.Q4Spec{}, fmt.Errorf("nil dense")
	}
	if l.Weights.Format != quant.FormatQ4_0 || l.Weights.Packed == nil {
		return fusedgpu.Q4Spec{}, fmt.Errorf("need Q4_0 packed (got %s)", l.Weights.Format.String())
	}
	scales, packed, ok := quant.Q4SIMD(l.Weights.Packed)
	if !ok {
		return fusedgpu.Q4Spec{}, fmt.Errorf("Q4 SIMD cache missing")
	}
	return fusedgpu.Q4Spec{
		Rows: l.Weights.Rows, Cols: l.Weights.Cols,
		Scales: scales, Packed: packed,
	}, nil
}

func rmsGammaF32(l *rmsnorm.Layer) ([]float32, error) {
	if l == nil || l.Gamma == nil {
		return nil, fmt.Errorf("nil rmsnorm")
	}
	return weights.MatrixF32(l.Gamma)
}

func onesF32(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = 1
	}
	return out
}
