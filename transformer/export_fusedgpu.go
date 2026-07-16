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
// Non-Q4 packed entities are unpacked once and re-packed to Q4_0 for upload
// (host entity stays in its native format; GPU always runs the Q4 fused path).
func (m *Model) ExportFusedGPUSpec() (*fusedgpu.Spec, error) {
	if m == nil {
		return nil, fmt.Errorf("transformer: nil model")
	}
	if !m.FusedGPUReady() {
		return nil, fmt.Errorf("transformer: need baked packed entity for gpu_fuse")
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
	// Cap KV footprint for fused GPU — host MaxSeqLen may be 2048+ for long-context
	// CPU paths; 4GB cards OOM on kc_* after several sequential SyncGPU uploads.
	const gpuMaxSeq = 256
	if maxSeq > gpuMaxSeq {
		maxSeq = gpuMaxSeq
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
		scales, packed, err := q4ViewFromBlob(blob)
		if err != nil {
			return fmt.Errorf("lm_head packed: %w", err)
		}
		spec.LMScales = scales
		spec.LMPacked = packed
		return nil
	}
	if head := m.UntiedLMHead(); head != nil && head.Packed != nil {
		scales, packed, err := q4ViewFromBlob(head.Packed)
		if err != nil {
			return fmt.Errorf("lm_head: %w", err)
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
	if l.Weights.Packed == nil {
		return fusedgpu.Q4Spec{}, fmt.Errorf("need packed weights (got %s)", l.Weights.Format.String())
	}
	scales, packed, err := q4ViewFromBlob(l.Weights.Packed)
	if err != nil {
		return fusedgpu.Q4Spec{}, err
	}
	return fusedgpu.Q4Spec{
		Rows: l.Weights.Rows, Cols: l.Weights.Cols,
		Scales: scales, Packed: packed,
	}, nil
}

// q4ViewFromBlob returns Q4_0 SIMD scales/packed, projecting other formats via Unpack→Pack.
// Does not retain F32Cache on the source blob — sequential gpu_fuse benches OOMs when every
// layer's inflate view stays live during CreateBufferInit staging.
func q4ViewFromBlob(b *quant.Blob) (scales []float32, packed []uint32, err error) {
	if b == nil {
		return nil, nil, fmt.Errorf("nil blob")
	}
	if b.Format == quant.FormatQ4_0 {
		quant.EnsureQ4SIMDCache(b)
		s, p, ok := quant.Q4SIMD(b)
		if !ok {
			return nil, nil, fmt.Errorf("Q4 SIMD cache missing")
		}
		return s, p, nil
	}
	need := b.Rows * b.Cols
	var f32 []float32
	if len(b.F32Cache) >= need {
		f32 = b.F32Cache[:need]
	} else {
		f32, err = quant.Unpack(b)
		if err != nil {
			return nil, nil, err
		}
	}
	q4, err := quant.Pack(quant.FormatQ4_0, f32[:need], b.Rows, b.Cols)
	if err != nil {
		return nil, nil, err
	}
	// Drop any inflate view left from simd_fuse so host RSS falls before GPU upload.
	b.F32Cache = nil
	quant.EnsureQ4SIMDCache(q4)
	s, p, ok := quant.Q4SIMD(q4)
	if !ok {
		return nil, nil, fmt.Errorf("Q4 project failed for %s", b.Format)
	}
	return s, p, nil
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
