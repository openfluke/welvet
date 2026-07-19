package transformer

import (
	"fmt"

	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/weights"
)

// PackWeights quantizes all dense projection weights for fused SIMD/GPU paths.
// Embedding table and RMSNorm γ stay Float32 (Lucy parity).
func (m *Model) PackWeights(format quant.Format) error {
	if m == nil {
		return fmt.Errorf("transformer: nil model")
	}
	if format == quant.FormatNone {
		return fmt.Errorf("fused pack: need a quant format (not none)")
	}
	packDense := func(l *dense.Layer, label string) error {
		if l == nil || l.Weights == nil {
			return nil
		}
		if err := l.Weights.Pack(format); err != nil {
			return fmt.Errorf("%s: %w", label, err)
		}
		return nil
	}
	for i, b := range m.Blocks {
		prefix := fmt.Sprintf("block %d", i)
		if b.Attn != nil {
			if err := packDense(b.Attn.Q, prefix+" Q"); err != nil {
				return err
			}
			if err := packDense(b.Attn.K, prefix+" K"); err != nil {
				return err
			}
			if err := packDense(b.Attn.V, prefix+" V"); err != nil {
				return err
			}
			if err := packDense(b.Attn.O, prefix+" O"); err != nil {
				return err
			}
		}
		if b.FFN != nil {
			if err := packDense(b.FFN.Gate, prefix+" gate"); err != nil {
				return err
			}
			if err := packDense(b.FFN.Up, prefix+" up"); err != nil {
				return err
			}
			if err := packDense(b.FFN.Down, prefix+" down"); err != nil {
				return err
			}
		}
	}
	if err := m.packLMHead(format); err != nil {
		return err
	}
	m.PackFormat = format
	m.FusedPack = true
	return nil
}

func (m *Model) packLMHead(format quant.Format) error {
	if m.VocabSize <= 0 || m.HiddenSize <= 0 {
		return fmt.Errorf("lm_head: bad dims")
	}
	n := m.VocabSize * m.HiddenSize
	if m.lmHead != nil {
		if err := m.lmHead.Pack(format); err != nil {
			return fmt.Errorf("lm_head pack: %w", err)
		}
		return nil
	}
	if m.Embed == nil || m.Embed.Weights == nil {
		return fmt.Errorf("lm_head: no weights")
	}
	w, err := weights.MatrixF32(m.Embed.Weights)
	if err != nil {
		return err
	}
	blob, err := quant.Pack(format, w[:n], m.VocabSize, m.HiddenSize)
	if err != nil {
		return fmt.Errorf("lm_head pack: %w", err)
	}
	m.lmHeadPacked = blob
	quant.EnsureQ4SIMDCache(blob)
	return nil
}

// EnsureFused prepares packed weights for simd_fuse / gpu_fuse profiles.
func (m *Model) EnsureFused(format quant.Format) error {
	if m == nil {
		return fmt.Errorf("transformer: nil model")
	}
	if m.FusedPack && m.PackFormat == format {
		return nil
	}
	fmt.Printf("  packing weights → %s (fused path)…\n", format.String())
	return m.PackWeights(format)
}
