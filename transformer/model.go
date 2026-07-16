// Package transformer runs Llama-style decoder generate from Welvet ENTITY packs.
//
// Tests live in w2a — not here.
package transformer

import (
	"github.com/openfluke/welvet/embedding"
	"github.com/openfluke/welvet/mha"
	"github.com/openfluke/welvet/rmsnorm"
	"github.com/openfluke/welvet/swiglu"
	"github.com/openfluke/welvet/weights"
)

// Block is one decoder layer: RMSNorm → MHA → residual → RMSNorm → SwiGLU → residual.
type Block struct {
	AttnNorm *rmsnorm.Layer
	Attn     *mha.Layer
	FFNNorm  *rmsnorm.Layer
	FFN      *swiglu.Layer
}

// Model is a causal LM loaded from .entity.
type Model struct {
	HiddenSize int
	VocabSize  int
	LMHeadTied bool
	HasFinalNorm bool
	MaxSeqLen  int
	EOSTokens  []int
	Repo       string
	Snapshot   string
	TokenizerPath string

	Embed     *embedding.Layer
	Blocks    []Block
	FinalNorm *rmsnorm.Layer

	// lmHead is vocab×hidden when untied; nil when tied (use Embed.Weights).
	lmHead *weights.Store
}
