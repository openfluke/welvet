package transformer

import (
	"fmt"

	"github.com/openfluke/welvet/dense"
	"github.com/openfluke/welvet/fusedgpu"
	"github.com/openfluke/welvet/quant"
)

// ExportHybridFusedGPUSpec builds the full BinaryG128 on-device hybrid decoder bundle.
// Every projection/embed/LM weight is included — no host GEMV fallback.
func (m *Model) ExportHybridFusedGPUSpec() (*fusedgpu.HybridSpec, error) {
	if m == nil {
		return nil, fmt.Errorf("transformer: nil model")
	}
	if !m.isHybrid() {
		return nil, fmt.Errorf("transformer: not a hybrid entity")
	}
	if len(m.Blocks) == 0 {
		return nil, fmt.Errorf("transformer: no blocks")
	}
	inter := 0
	if m.Blocks[0].FFN != nil {
		inter = m.Blocks[0].FFN.Cfg.IntermediateDim
	}
	if inter <= 0 {
		return nil, fmt.Errorf("transformer: intermediate size unset")
	}
	eps := float32(1e-6)
	if m.Blocks[0].AttnNorm != nil && m.Blocks[0].AttnNorm.Cfg.Eps > 0 {
		eps = float32(m.Blocks[0].AttnNorm.Cfg.Eps)
	}
	maxSeq := m.MaxSeqLen
	if maxSeq <= 0 {
		maxSeq = 256
	}
	const gpuMaxSeq = 512
	if maxSeq > gpuMaxSeq {
		maxSeq = gpuMaxSeq
	}

	embed, err := binarySpecFromBlob(m.embedPacked)
	if err != nil {
		return nil, fmt.Errorf("embed: %w", err)
	}
	lmBlob := m.lmHeadPacked
	if lmBlob == nil && m.lmHead != nil {
		lmBlob = m.lmHead.Packed
	}
	if lmBlob == nil {
		lmBlob = m.embedPacked // tied
	}
	lm, err := binarySpecFromBlob(lmBlob)
	if err != nil {
		return nil, fmt.Errorf("lm_head: %w", err)
	}

	spec := &fusedgpu.HybridSpec{
		Hidden:       m.HiddenSize,
		Vocab:        m.VocabSize,
		Layers:       len(m.Blocks),
		Intermediate: inter,
		Eps:          eps,
		MaxSeq:       maxSeq,
		Embed:        embed,
		LMHead:       lm,
		Blocks:       make([]fusedgpu.HybridBlockSpec, len(m.Blocks)),
	}
	if m.HasFinalNorm && m.FinalNorm != nil {
		spec.FinalNorm, err = rmsGammaF32(m.FinalNorm)
		if err != nil {
			return nil, fmt.Errorf("final_norm: %w", err)
		}
	} else {
		spec.FinalNorm = onesF32(m.HiddenSize)
	}

	for i := range m.Blocks {
		b := &m.Blocks[i]
		blk := &spec.Blocks[i]
		blk.LayerType = b.LayerType
		if b.AttnNorm != nil {
			blk.AttnNorm, err = rmsGammaF32(b.AttnNorm)
			if err != nil {
				return nil, fmt.Errorf("block %d attn_norm: %w", i, err)
			}
		}
		if b.FFNNorm != nil {
			blk.FFNNorm, err = rmsGammaF32(b.FFNNorm)
			if err != nil {
				return nil, fmt.Errorf("block %d ffn_norm: %w", i, err)
			}
		}
		if blk.Gate, err = binarySpecFromDense(b.FFN.Gate); err != nil {
			return nil, fmt.Errorf("block %d gate: %w", i, err)
		}
		if blk.Up, err = binarySpecFromDense(b.FFN.Up); err != nil {
			return nil, fmt.Errorf("block %d up: %w", i, err)
		}
		if blk.Down, err = binarySpecFromDense(b.FFN.Down); err != nil {
			return nil, fmt.Errorf("block %d down: %w", i, err)
		}

		switch b.LayerType {
		case "full_attention":
			if blk.Q, err = binarySpecFromDense(b.Q); err != nil {
				return nil, fmt.Errorf("block %d q: %w", i, err)
			}
			if blk.K, err = binarySpecFromDense(b.K); err != nil {
				return nil, fmt.Errorf("block %d k: %w", i, err)
			}
			if blk.V, err = binarySpecFromDense(b.V); err != nil {
				return nil, fmt.Errorf("block %d v: %w", i, err)
			}
			if blk.O, err = binarySpecFromDense(b.O); err != nil {
				return nil, fmt.Errorf("block %d o: %w", i, err)
			}
			blk.QNorm = append([]float32(nil), b.QNorm...)
			blk.KNorm = append([]float32(nil), b.KNorm...)
			blk.OutputGate = b.OutputGate
			blk.PartialRotary = float32(b.PartialRotary)
			blk.RoPETheta = float32(b.RoPETheta)
			if blk.RoPETheta <= 0 {
				blk.RoPETheta = 10000
			}
			blk.NumHeads = b.NumHeads
			blk.NumKVHeads = b.NumKVHeads
			blk.HeadDim = b.HeadDim
		case "linear_attention":
			g := b.GDN
			if g == nil {
				return nil, fmt.Errorf("block %d: nil GDN", i)
			}
			if blk.GDNQKV, err = binarySpecFromBlob(g.InQKV); err != nil {
				return nil, fmt.Errorf("block %d gdn_qkv: %w", i, err)
			}
			if blk.GDNZ, err = binarySpecFromBlob(g.InZ); err != nil {
				return nil, fmt.Errorf("block %d gdn_z: %w", i, err)
			}
			if blk.GDNB, err = binarySpecFromBlob(g.InB); err != nil {
				return nil, fmt.Errorf("block %d gdn_b: %w", i, err)
			}
			if blk.GDNA, err = binarySpecFromBlob(g.InA); err != nil {
				return nil, fmt.Errorf("block %d gdn_a: %w", i, err)
			}
			if blk.GDNOut, err = binarySpecFromBlob(g.Out); err != nil {
				return nil, fmt.Errorf("block %d gdn_out: %w", i, err)
			}
			blk.GDNConv = append([]float32(nil), g.ConvWeight...)
			blk.GDNALog = append([]float32(nil), g.ALog...)
			blk.GDNDtBias = append([]float32(nil), g.DtBias...)
			blk.GDNNorm = append([]float32(nil), g.NormGamma...)
			blk.NumKeyHeads = g.Cfg.NumKeyHeads
			blk.NumValueHeads = g.Cfg.NumValueHeads
			blk.KeyHeadDim = g.Cfg.KeyHeadDim
			blk.ValueHeadDim = g.Cfg.ValueHeadDim
			blk.ConvKernel = g.Cfg.ConvKernel
		default:
			return nil, fmt.Errorf("block %d: unknown layer type %q", i, b.LayerType)
		}
	}
	return spec, nil
}

func binarySpecFromDense(l *dense.Layer) (fusedgpu.BinarySpec, error) {
	if l == nil || l.Weights == nil || l.Weights.Packed == nil {
		return fusedgpu.BinarySpec{}, fmt.Errorf("nil packed dense")
	}
	return binarySpecFromBlob(l.Weights.Packed)
}

func binarySpecFromBlob(b *quant.Blob) (fusedgpu.BinarySpec, error) {
	if b == nil {
		return fusedgpu.BinarySpec{}, fmt.Errorf("nil blob")
	}
	if b.Format != quant.FormatBinaryPacked {
		return fusedgpu.BinarySpec{}, fmt.Errorf("want BinaryPacked got %s", b.Format.String())
	}
	scales, words, g128, ok := dense.BinaryBlobStaging(b)
	if !ok || !g128 {
		return fusedgpu.BinarySpec{}, fmt.Errorf("BinaryG128 staging failed (%dx%d)", b.Rows, b.Cols)
	}
	return fusedgpu.BinarySpec{
		Rows:   b.Rows,
		Cols:   b.Cols,
		Scales: scales,
		Words:  words,
	}, nil
}
