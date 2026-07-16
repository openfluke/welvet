package transformer

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/embedding"
	"github.com/openfluke/welvet/mha"
	"github.com/openfluke/welvet/rmsnorm"
	"github.com/openfluke/welvet/swiglu"
	"github.com/openfluke/welvet/weights"
)

// ForwardTokens runs the decoder on token IDs and returns last-position logits.
// Prefill with the full prompt once, then call with a single new token for decode
// so MHA KV cache stays warm.
func (m *Model) ForwardTokens(ids []uint32) ([]float32, error) {
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

	h := emb
	for i, b := range m.Blocks {
		_, n1, err := rmsnorm.Forward(b.AttnNorm, h)
		if err != nil {
			return nil, fmt.Errorf("block %d attn_norm: %w", i, err)
		}
		_, a, err := mha.Forward(b.Attn, n1)
		if err != nil {
			return nil, fmt.Errorf("block %d attn: %w", i, err)
		}
		h = residualAdd(h, a)

		_, n2, err := rmsnorm.Forward(b.FFNNorm, h)
		if err != nil {
			return nil, fmt.Errorf("block %d ffn_norm: %w", i, err)
		}
		_, f, err := swiglu.Forward(b.FFN, n2)
		if err != nil {
			return nil, fmt.Errorf("block %d ffn: %w", i, err)
		}
		h = residualAdd(h, f)
	}

	if m.HasFinalNorm && m.FinalNorm != nil {
		_, h, err = rmsnorm.Forward(m.FinalNorm, h)
		if err != nil {
			return nil, fmt.Errorf("final_norm: %w", err)
		}
	}

	last := lastRow(h, m.HiddenSize)
	return m.applyLMHead(last)
}

func residualAdd(a, b *core.Tensor[float32]) *core.Tensor[float32] {
	out := core.NewTensor[float32](a.Shape...)
	n := len(a.Data)
	if len(b.Data) < n {
		n = len(b.Data)
	}
	for i := 0; i < n; i++ {
		out.Data[i] = a.Data[i] + b.Data[i]
	}
	return out
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
	var store *weights.Store
	if m.lmHead != nil {
		store = m.lmHead
	} else if m.Embed != nil {
		store = m.Embed.Weights
	} else {
		return nil, fmt.Errorf("lm_head: no weights")
	}
	if err := weights.MatVec(store, hidden, logits); err != nil {
		return nil, err
	}
	return logits, nil
}
