package transformer

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/dense"
	"github.com/openfluke/welvet/embedding"
	"github.com/openfluke/welvet/mha"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/rmsnorm"
	"github.com/openfluke/welvet/swiglu"
	"github.com/openfluke/welvet/weights"
)

// ForwardTokens runs the decoder on token IDs and returns last-position logits.
// Prefill with the full prompt once, then call with a single new token for decode
// so MHA KV cache stays warm.
func (m *Model) ForwardTokens(ids []uint32) ([]float32, error) {
	if m.gpu != nil {
		return m.ForwardTokensGPU(ids)
	}
	return m.forwardTokensHost(ids)
}

func (m *Model) forwardTokensHost(ids []uint32) ([]float32, error) {
	if m == nil || m.Embed == nil {
		return nil, fmt.Errorf("transformer: nil model")
	}
	if len(ids) == 0 {
		return nil, fmt.Errorf("transformer: empty ids")
	}
	if len(ids) > m.MaxSeqLen {
		return nil, fmt.Errorf("transformer: seq %d > MaxSeqLen %d", len(ids), m.MaxSeqLen)
	}

	tok := core.NewTensor[float32](1, len(ids))
	for i, id := range ids {
		tok.Data[i] = float32(id)
	}
	_, emb, err := embedding.Forward(m.Embed, tok)
	if err != nil {
		return nil, fmt.Errorf("embed: %w", err)
	}

	sc := m.ensureScratch(emb.Shape, len(emb.Data))
	copy(sc.h.Data[:len(emb.Data)], emb.Data)
	h := sc.h

	for i, b := range m.Blocks {
		_, n1, err := rmsnorm.Forward(b.AttnNorm, h)
		if err != nil {
			return nil, fmt.Errorf("block %d attn_norm: %w", i, err)
		}
		_, a, err := mha.Forward(b.Attn, n1)
		if err != nil {
			return nil, fmt.Errorf("block %d attn: %w", i, err)
		}
		residualAddInPlace(h, a)

		_, n2, err := rmsnorm.Forward(b.FFNNorm, h)
		if err != nil {
			return nil, fmt.Errorf("block %d ffn_norm: %w", i, err)
		}
		_, f, err := swiglu.Forward(b.FFN, n2)
		if err != nil {
			return nil, fmt.Errorf("block %d ffn: %w", i, err)
		}
		residualAddInPlace(h, f)
	}

	if m.HasFinalNorm && m.FinalNorm != nil {
		_, normed, err := rmsnorm.Forward(m.FinalNorm, h)
		if err != nil {
			return nil, fmt.Errorf("final_norm: %w", err)
		}
		copy(h.Data[:len(normed.Data)], normed.Data)
	}

	off := len(h.Data) - m.HiddenSize
	if off < 0 {
		return nil, fmt.Errorf("transformer: hidden short")
	}
	return m.applyLMHead(h.Data[off:])
}

func lastRow(t *core.Tensor[float32], hidden int) []float32 {
	if t == nil || hidden <= 0 || len(t.Data) < hidden {
		return nil
	}
	off := len(t.Data) - hidden
	row := make([]float32, hidden)
	copy(row, t.Data[off:])
	return row
}

func (m *Model) applyLMHead(hidden []float32) ([]float32, error) {
	logits := make([]float32, m.VocabSize)
	store, err := m.lmHeadStore()
	if err != nil {
		return nil, err
	}
	if m.Exec.Backend == core.BackendWebGPU && m.gpu == nil {
		if err := dense.MatVecWebGPU(store, hidden, logits, 1, m.HiddenSize, m.VocabSize); err != nil {
			return nil, fmt.Errorf("lm_head gpu: %w", err)
		}
		return logits, nil
	}
	if store.Format == quant.FormatQ4_0 && store.Packed != nil {
		if err := dense.MatVecQ4_0Blob(store.Packed, hidden, logits); err != nil {
			return nil, fmt.Errorf("lm_head q4: %w", err)
		}
		return logits, nil
	}
	if store.Format != quant.FormatNone && store.Packed != nil {
		if err := weights.MatVec(store, hidden, logits); err != nil {
			return nil, fmt.Errorf("lm_head packed: %w", err)
		}
		return logits, nil
	}
	if err := weights.MatVec(store, hidden, logits); err != nil {
		return nil, fmt.Errorf("lm_head: %w", err)
	}
	return logits, nil
}

func (m *Model) lmHeadStore() (*weights.Store, error) {
	if m.lmHead != nil {
		return m.lmHead, nil
	}
	if m.lmHeadPacked != nil {
		return &weights.Store{
			Rows:   m.VocabSize,
			Cols:   m.HiddenSize,
			Format: m.PackFormat,
			Packed: m.lmHeadPacked,
			DType:  core.DTypeFloat32,
		}, nil
	}
	if m.Embed != nil && m.Embed.Weights != nil {
		return m.Embed.Weights, nil
	}
	return nil, fmt.Errorf("lm_head: no weights")
}
