package fusedgpu

import (
	"fmt"
)

// Spec is the CPU-side weight bundle for the fused GPU engine.
type Spec struct {
	Hidden, Vocab, Layers int
	Heads, KVHeads          int
	HeadDim, QDim, KVDim    int
	Intermediate          int
	Eps                   float32
	RopeTheta             float32
	MaxSeq                int

	Embed     []float32
	FinalNorm []float32
	LMScales  []float32
	LMPacked  []uint32
	Blocks    []BlockSpec
}

// BlockSpec is one decoder layer's weights.
type BlockSpec struct {
	AttnNorm []float32
	MLPNorm  []float32
	Q, K, V, O Q4Spec
	Gate, Up, Down Q4Spec
}

// Q4Spec is one Q4_0 matrix in GPU layout.
type Q4Spec struct {
	Rows, Cols int
	Scales     []float32
	Packed     []uint32
}

// NewFromSpec builds and uploads the fused GPU engine.
func NewFromSpec(spec *Spec) (*Engine, error) {
	if spec == nil {
		return nil, fmt.Errorf("fusedgpu: nil spec")
	}
	cm, err := spec.toModelCPU()
	if err != nil {
		return nil, err
	}
	// Spec slices were moved into cm — drop leftovers so Export temps can GC
	// before CreateBufferInit staging allocates another full copy.
	spec.clearPayloads()
	e, err := newEngine(cm)
	if err != nil {
		return nil, err
	}
	return &Engine{e: e}, nil
}

func (s *Spec) clearPayloads() {
	if s == nil {
		return
	}
	s.Embed, s.FinalNorm, s.LMScales, s.LMPacked = nil, nil, nil, nil
	for i := range s.Blocks {
		b := &s.Blocks[i]
		b.AttnNorm, b.MLPNorm = nil, nil
		b.Q.Scales, b.Q.Packed = nil, nil
		b.K.Scales, b.K.Packed = nil, nil
		b.V.Scales, b.V.Packed = nil, nil
		b.O.Scales, b.O.Packed = nil, nil
		b.Gate.Scales, b.Gate.Packed = nil, nil
		b.Up.Scales, b.Up.Packed = nil, nil
		b.Down.Scales, b.Down.Packed = nil, nil
	}
	s.Blocks = nil
}

func (s *Spec) toModelCPU() (*modelCPU, error) {
	if s.Layers <= 0 || len(s.Blocks) != s.Layers {
		return nil, fmt.Errorf("fusedgpu: block count mismatch")
	}
	// Take ownership of slices (no deep copy) — upload then dropHostWeightPayloads.
	m := &modelCPU{
		hidden: s.Hidden, vocab: s.Vocab, layers: s.Layers,
		heads: s.Heads, kvHeads: s.KVHeads, headDim: s.HeadDim,
		qDim: s.QDim, kvDim: s.KVDim, intermediate: s.Intermediate,
		eps: s.Eps, ropeTheta: s.RopeTheta, maxSeq: s.MaxSeq,
		embed: s.Embed, finalNorm: s.FinalNorm,
		lmScales: s.LMScales, lmPacked: s.LMPacked,
	}
	m.blocks = make([]blockCPU, s.Layers)
	for i, b := range s.Blocks {
		m.blocks[i] = blockCPU{
			attnNorm: rmsW{w: b.AttnNorm},
			mlpNorm:  rmsW{w: b.MLPNorm},
			q:        b.Q.toMat(), k: b.K.toMat(), v: b.V.toMat(), o: b.O.toMat(),
			gate: b.Gate.toMat(), up: b.Up.toMat(), down: b.Down.toMat(),
		}
	}
	return m, nil
}

func (q Q4Spec) toMat() q4Mat {
	return q4Mat{rows: q.Rows, cols: q.Cols, scales: q.Scales, packed: q.Packed}
}
