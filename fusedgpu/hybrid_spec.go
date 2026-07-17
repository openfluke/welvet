package fusedgpu

import "fmt"

// BinarySpec is one BinaryG128 matrix (native MLX layout: 1 scale / 128 weights).
type BinarySpec struct {
	Rows, Cols int
	Scales     []float32
	Words      []uint32
}

// HybridBlockSpec is one Qwen3.5 / Bonsai decoder layer.
type HybridBlockSpec struct {
	LayerType string // "linear_attention" | "full_attention"
	AttnNorm  []float32
	FFNNorm   []float32
	Gate, Up, Down BinarySpec

	// full_attention
	Q, K, V, O BinarySpec
	QNorm, KNorm []float32
	OutputGate   bool
	PartialRotary float32
	RoPETheta     float32
	NumHeads, NumKVHeads, HeadDim int

	// linear_attention (GDN)
	GDNQKV, GDNZ, GDNB, GDNA, GDNOut BinarySpec
	GDNConv                           []float32
	GDNALog, GDNDtBias                []float32
	GDNNorm                           []float32
	NumKeyHeads, NumValueHeads        int
	KeyHeadDim, ValueHeadDim          int
	ConvKernel                        int
}

// HybridSpec is the full on-device BinaryG128 hybrid decoder bundle.
type HybridSpec struct {
	Hidden, Vocab, Layers int
	Intermediate          int
	Eps                   float32
	MaxSeq                int
	LMHeadTied            bool // reuse embed GPU buffers for LM head
	Embed                 BinarySpec // vocab × hidden
	FinalNorm             []float32
	LMHead                BinarySpec // vocab × hidden (empty when LMHeadTied)
	Blocks                []HybridBlockSpec
}

// HybridEngine is the full BinaryG128 hybrid decoder on WebGPU (no host GEMV).
type HybridEngine struct {
	e *hybridEngine
}

// NewHybridFromSpec uploads every weight to GPU and builds the fused hybrid engine.
// Requires enough VRAM for the full BinaryPacked working set (~entity size + scratch).
func NewHybridFromSpec(spec *HybridSpec) (*HybridEngine, error) {
	if spec == nil {
		return nil, fmt.Errorf("fusedgpu: nil hybrid spec")
	}
	if spec.Layers <= 0 || len(spec.Blocks) != spec.Layers {
		return nil, fmt.Errorf("fusedgpu: hybrid block count mismatch")
	}
	if spec.Intermediate <= 0 {
		return nil, fmt.Errorf("fusedgpu: hybrid Intermediate unset")
	}
	e, err := newHybridEngine(spec)
	if err != nil {
		return nil, err
	}
	spec.clearPayloads()
	return &HybridEngine{e: e}, nil
}

// Close releases GPU resources for this engine (shared device kept).
func (eng *HybridEngine) Close() {
	if eng == nil || eng.e == nil {
		return
	}
	eng.e.release()
	eng.e = nil
}

// Reset clears GDN/KV state and position for a new prompt.
func (eng *HybridEngine) Reset() error {
	if eng == nil || eng.e == nil {
		return fmt.Errorf("fusedgpu: nil hybrid engine")
	}
	return eng.e.resetState()
}

// AppendTokens runs one or more forward steps; returns logits for the last token.
func (eng *HybridEngine) AppendTokens(ids []uint32) ([]float32, error) {
	if eng == nil || eng.e == nil {
		return nil, fmt.Errorf("fusedgpu: nil hybrid engine")
	}
	return eng.e.appendTokens(ids)
}

// PrefillSample runs the prompt on-device and returns the greedy next token (4-byte readback).
func (eng *HybridEngine) PrefillSample(ids []uint32) (uint32, error) {
	if eng == nil || eng.e == nil {
		return 0, fmt.Errorf("fusedgpu: nil hybrid engine")
	}
	return eng.e.prefillSample(ids)
}

// DecodeSample embeds tok, runs one decode step, returns the next greedy token (4-byte readback).
func (eng *HybridEngine) DecodeSample(tok uint32) (uint32, error) {
	if eng == nil || eng.e == nil {
		return 0, fmt.Errorf("fusedgpu: nil hybrid engine")
	}
	return eng.e.stepTokenSample(tok)
}

// DecodeChunk runs k decode steps in one GPU submit (one MapAsync for k tokens).
// Requires GPU token/pos already set (after PrefillSample or a prior chunk).
func (eng *HybridEngine) DecodeChunk(k int) ([]uint32, error) {
	if eng == nil || eng.e == nil {
		return nil, fmt.Errorf("fusedgpu: nil hybrid engine")
	}
	return eng.e.decodeChunkSample(k)
}

// Pos returns the current sequence position (prompt tokens processed + decode advances).
func (eng *HybridEngine) Pos() int {
	if eng == nil || eng.e == nil {
		return 0
	}
	return eng.e.pos
}

// MaxSeq returns the engine context limit.
func (eng *HybridEngine) MaxSeq() int {
	if eng == nil || eng.e == nil {
		return 0
	}
	return eng.e.maxSeq
}

// AdapterName returns the bound GPU adapter name.
func (eng *HybridEngine) AdapterName() string {
	if eng == nil || eng.e == nil || eng.e.adapter == nil {
		return ""
	}
	return eng.e.adapter.GetInfo().Name
}

// VRAMBytes returns allocated WebGPU buffer bytes for this hybrid engine.
func (eng *HybridEngine) VRAMBytes() uint64 {
	if eng == nil || eng.e == nil {
		return 0
	}
	return eng.e.estimateVRAM()
}

func (s *BinarySpec) nbytes() uint64 {
	if s == nil {
		return 0
	}
	return uint64(len(s.Scales)*4 + len(s.Words)*4)
}

func (s *HybridSpec) clearPayloads() {
	if s == nil {
		return
	}
	s.Embed.Scales, s.Embed.Words = nil, nil
	s.FinalNorm = nil
	s.LMHead.Scales, s.LMHead.Words = nil, nil
	for i := range s.Blocks {
		b := &s.Blocks[i]
		b.AttnNorm, b.FFNNorm = nil, nil
		b.Gate.Scales, b.Gate.Words = nil, nil
		b.Up.Scales, b.Up.Words = nil, nil
		b.Down.Scales, b.Down.Words = nil, nil
		b.Q.Scales, b.Q.Words = nil, nil
		b.K.Scales, b.K.Words = nil, nil
		b.V.Scales, b.V.Words = nil, nil
		b.O.Scales, b.O.Words = nil, nil
		b.QNorm, b.KNorm = nil, nil
		b.GDNQKV.Scales, b.GDNQKV.Words = nil, nil
		b.GDNZ.Scales, b.GDNZ.Words = nil, nil
		b.GDNB.Scales, b.GDNB.Words = nil, nil
		b.GDNA.Scales, b.GDNA.Words = nil, nil
		b.GDNOut.Scales, b.GDNOut.Words = nil, nil
		b.GDNConv, b.GDNALog, b.GDNDtBias, b.GDNNorm = nil, nil, nil, nil
	}
	s.Blocks = nil
}
