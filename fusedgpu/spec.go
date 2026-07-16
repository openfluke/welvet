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
	e, err := newEngine(cm)
	if err != nil {
		return nil, err
	}
	return &Engine{e: e}, nil
}

func (s *Spec) toModelCPU() (*modelCPU, error) {
	if s.Layers <= 0 || len(s.Blocks) != s.Layers {
		return nil, fmt.Errorf("fusedgpu: block count mismatch")
	}
	m := &modelCPU{
		hidden: s.Hidden, vocab: s.Vocab, layers: s.Layers,
		heads: s.Heads, kvHeads: s.KVHeads, headDim: s.HeadDim,
		qDim: s.QDim, kvDim: s.KVDim, intermediate: s.Intermediate,
		eps: s.Eps, ropeTheta: s.RopeTheta, maxSeq: s.MaxSeq,
		embed: append([]float32(nil), s.Embed...),
		finalNorm: append([]float32(nil), s.FinalNorm...),
		lmScales: append([]float32(nil), s.LMScales...),
		lmPacked: append([]uint32(nil), s.LMPacked...),
	}
	m.blocks = make([]blockCPU, s.Layers)
	for i, b := range s.Blocks {
		m.blocks[i] = blockCPU{
			attnNorm: rmsW{w: append([]float32(nil), b.AttnNorm...)},
			mlpNorm:  rmsW{w: append([]float32(nil), b.MLPNorm...)},
			q:        b.Q.toMat(), k: b.K.toMat(), v: b.V.toMat(), o: b.O.toMat(),
			gate: b.Gate.toMat(), up: b.Up.toMat(), down: b.Down.toMat(),
		}
	}
	return m, nil
}

func (q Q4Spec) toMat() q4Mat {
	return q4Mat{rows: q.Rows, cols: q.Cols, scales: q.Scales, packed: q.Packed}
}
