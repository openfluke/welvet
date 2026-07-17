// Package transformer runs Llama-style decoder generate from Welvet ENTITY packs.
//
// Tests live in w2a — not here.
package transformer

import (
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/dense"
	"github.com/openfluke/welvet/embedding"
	"github.com/openfluke/welvet/gdn"
	"github.com/openfluke/welvet/mha"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/rmsnorm"
	"github.com/openfluke/welvet/swiglu"
	"github.com/openfluke/welvet/weights"
)

// Block is one decoder layer: RMSNorm → mixer → residual → RMSNorm → SwiGLU → residual.
type Block struct {
	AttnNorm *rmsnorm.Layer
	Attn     *mha.Layer // Llama-style path
	FFNNorm  *rmsnorm.Layer
	FFN      *swiglu.Layer

	// Qwen3.5 / Bonsai hybrid fields (optional)
	LayerType     string
	GDN           *gdn.Layer
	Q, K, V, O    *dense.Layer
	QNorm, KNorm  []float32
	NumHeads      int
	NumKVHeads    int
	HeadDim       int
	RoPETheta     float64
	PartialRotary float64
	OutputGate    bool
	KVCacheK      []float32
	KVCacheV      []float32
	KVOffset      int
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
	// EntityPath is the source .entity (used to drop/reload host weights after GPU fuse).
	EntityPath    string
	Architecture  string
	LayerTypes    []string
	AttnOutputGate bool
	PartialRotary  float64

	Embed     *embedding.Layer
	embedPacked *quant.Blob // Bonsai / MLX 1-bit embed table
	Blocks    []Block
	FinalNorm *rmsnorm.Layer

	// lmHead is vocab×hidden when untied; nil when tied (use Embed.Weights).
	lmHead *weights.Store
	// lmHeadPacked is tied-head packed logits matrix (embed table stays F32).
	lmHeadPacked *quant.Blob

	// Exec is the active generate backend (set by ApplyExec).
	Exec core.ExecConfig
	// Fused enables packed-quant fused matmul paths (simd_fuse / gpu_fuse).
	Fused bool
	// PackFormat is the active fused quant layout (all k-quants / IQ / BitNet).
	PackFormat quant.Format
	FusedPack  bool

	// gpu holds *fusedgpu.Engine when gpu_fuse profile synced (see gpu.go).
	gpu any
	// hostWeightsReleased: host packed Raw/Scales were dropped after GPU upload (anti double-mount).
	hostWeightsReleased bool

	scratch       *fwdScratch
	logitsScratch []float32 // reused LM-head output
	// Quiet suppresses hybrid prefill progress lines (set by Generate when Silent).
	Quiet bool
}

// FusedGPUReady reports whether the Q4 monolithic fused GPU decoder can run.
// Hybrid Qwen3.5 / Bonsai uses HybridEngine (SyncHybridFused) instead.
func (m *Model) FusedGPUReady() bool {
	if m == nil || !m.FusedPack || m.PackFormat == quant.FormatNone {
		return false
	}
	if m.isHybrid() {
		return false
	}
	return true
}

// HybridGPUFuse reports gpu_fuse on a Qwen3.5/Bonsai entity (full BinaryG128 fuse).
func (m *Model) HybridGPUFuse() bool {
	return m != nil && m.isHybrid() && m.Exec.Backend == core.BackendWebGPU && m.Fused
}

// LMHeadPackedBlob returns baked tied-head logits matrix when present.
func (m *Model) LMHeadPackedBlob() *quant.Blob { return m.lmHeadPacked }

// UntiedLMHead returns untied lm_head weights when present.
func (m *Model) UntiedLMHead() *weights.Store { return m.lmHead }
