package hf

import (
	"fmt"
	"strings"
)

// GlobalWeights are embeddings / optional untied LM head / final RMSNorm γ.
type GlobalWeights struct {
	Embeddings   []float32
	LMHead       []float32 // may alias Embeddings when tied
	FinalNorm    []float32
	HasFinalNorm bool
	LMHeadTied   bool
}

// MapGlobals finds embeddings, lm_head, final norm in a tensor map.
func MapGlobals(tensors map[string][]float32) GlobalWeights {
	emb := findTensor(tensors, []string{
		"model.embed_tokens.weight",
		"transformer.wte.weight",
		"embeddings.weight",
		"embed_tokens.weight",
	})
	fn := findTensor(tensors, []string{
		"model.norm.weight",
		"transformer.ln_f.weight",
		"ln_f.weight",
		"norm.weight",
	})
	lm := findTensor(tensors, []string{
		"lm_head.weight",
		"output.weight",
	})
	tied := false
	if lm == nil {
		lm = emb
		tied = true
	} else if len(emb) > 0 && len(lm) == len(emb) && &emb[0] == &lm[0] {
		tied = true
	}
	return GlobalWeights{
		Embeddings:   emb,
		LMHead:       lm,
		FinalNorm:    fn,
		HasFinalNorm: fn != nil,
		LMHeadTied:   tied,
	}
}

func findTensor(tensors map[string][]float32, patterns []string) []float32 {
	for _, pattern := range patterns {
		if t, ok := tensors[pattern]; ok {
			return t
		}
		for k, v := range tensors {
			if strings.HasSuffix(k, pattern) {
				return v
			}
		}
	}
	return nil
}

// BlockWeights holds one transformer block's Float32 tensors.
type BlockWeights struct {
	AttnNorm []float32
	Q, K, V, O []float32
	FFNNorm  []float32
	Gate, Up, Down []float32
}

// RouteBlockKey classifies a HF tensor name into block index + role.
// Roles: attn_norm, q, k, v, o, ffn_norm, gate, up, down.
func RouteBlockKey(key string) (layerIdx int, role string, ok bool) {
	layerIdx, ok = WeightLayerIndex(key)
	if !ok {
		return 0, "", false
	}
	switch {
	case strings.Contains(key, "input_layernorm") || strings.Contains(key, "ln_1"):
		return layerIdx, "attn_norm", true
	case strings.Contains(key, "post_attention_layernorm") || strings.Contains(key, "ln_2"):
		return layerIdx, "ffn_norm", true
	case strings.Contains(key, "q_proj"):
		return layerIdx, "q", true
	case strings.Contains(key, "k_proj"):
		return layerIdx, "k", true
	case strings.Contains(key, "v_proj"):
		return layerIdx, "v", true
	case strings.Contains(key, "o_proj"):
		return layerIdx, "o", true
	case strings.Contains(key, "gate_proj") || strings.Contains(key, "w1"):
		return layerIdx, "gate", true
	case strings.Contains(key, "up_proj") || strings.Contains(key, "w3"):
		return layerIdx, "up", true
	case strings.Contains(key, "down_proj") || strings.Contains(key, "w2"):
		return layerIdx, "down", true
	default:
		return 0, "", false
	}
}

// MapBlock fills BlockWeights from a selective tensor map for one layer.
func MapBlock(tensors map[string][]float32, layerIdx int) (BlockWeights, error) {
	var b BlockWeights
	for k, v := range tensors {
		li, role, ok := RouteBlockKey(k)
		if !ok || li != layerIdx {
			continue
		}
		switch role {
		case "attn_norm":
			b.AttnNorm = v
		case "ffn_norm":
			b.FFNNorm = v
		case "q":
			b.Q = v
		case "k":
			b.K = v
		case "v":
			b.V = v
		case "o":
			b.O = v
		case "gate":
			b.Gate = v
		case "up":
			b.Up = v
		case "down":
			b.Down = v
		}
	}
	if b.AttnNorm == nil || b.FFNNorm == nil ||
		b.Q == nil || b.K == nil || b.V == nil || b.O == nil ||
		b.Gate == nil || b.Up == nil || b.Down == nil {
		return b, fmt.Errorf("layer %d incomplete weights (missing q/k/v/o/norms/mlp)", layerIdx)
	}
	return b, nil
}

// CloneFloat32 copies a slice (nil-safe).
func CloneFloat32(src []float32) []float32 {
	if src == nil {
		return nil
	}
	return append([]float32(nil), src...)
}

// CloneGlobals copies globals so the temporary safetensors map can be freed.
func CloneGlobals(g GlobalWeights) GlobalWeights {
	out := GlobalWeights{
		HasFinalNorm: g.HasFinalNorm,
		LMHeadTied:   g.LMHeadTied,
		Embeddings:   CloneFloat32(g.Embeddings),
		FinalNorm:    CloneFloat32(g.FinalNorm),
	}
	if g.LMHeadTied {
		out.LMHead = out.Embeddings
	} else {
		out.LMHead = CloneFloat32(g.LMHead)
	}
	return out
}
