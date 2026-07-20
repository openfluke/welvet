package fountain

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/stub/seed"
)

// Config for NeuralFountain (dense specialists).
type Config struct {
	K           int
	Epochs      int
	LR          float64
	LossRate    float64
	MaxOverhead float64
	Seed        uint64
}

// DefaultConfig returns sane neural-fountain defaults.
func DefaultConfig() Config {
	return Config{K: 4, Epochs: 2, LR: 1e-2, LossRate: 0.1, MaxOverhead: 2, Seed: 1}
}

// Factory builds specialist i as a Grid.
type Factory func(i int) (*architecture.Grid, error)

// DenseFactory builds Dim→Dim Dense specialists.
func DenseFactory(dim int) Factory {
	return func(i int) (*architecture.Grid, error) {
		w := make([]float32, dim*dim)
		seed.InitFloat32He(w, dim, seed.DeriveLayer(uint64(i+1), 0, "fountain.dense"))
		l, err := dense.NewConfigured(dim, dim, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, w)
		if err != nil {
			return nil, err
		}
		g := architecture.NewGrid(1, 1, 1, 1)
		return g, dense.Place(g, 0, 0, 0, 0, l)
	}
}

// Master holds recovered specialist weight blobs after fountain transport.
type Master struct {
	Blobs [][]byte
	Dim   int
	K     int
}

// PackGridWeights concatenates Dense FlattenF32 as LE bytes.
func PackGridWeights(g *architecture.Grid) ([]byte, error) {
	if g == nil {
		return nil, fmt.Errorf("fountain: nil grid")
	}
	var out []byte
	for _, c := range g.HopOrder() {
		cell := g.At(c.Z, c.Y, c.X, c.L)
		if cell == nil {
			continue
		}
		op, ok := cell.Op.(*dense.Layer)
		if !ok || op.Weights == nil {
			continue
		}
		w, err := op.Weights.FlattenF32()
		if err != nil {
			return nil, err
		}
		raw := make([]byte, len(w)*4)
		for i, v := range w {
			binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
		}
		out = append(out, raw...)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("fountain: no dense weights")
	}
	return out, nil
}

// UnpackGridWeights writes LE float32 bytes into Dense stores (same layout as Pack).
func UnpackGridWeights(g *architecture.Grid, blob []byte) error {
	off := 0
	for _, c := range g.HopOrder() {
		cell := g.At(c.Z, c.Y, c.X, c.L)
		if cell == nil {
			continue
		}
		op, ok := cell.Op.(*dense.Layer)
		if !ok || op.Weights == nil {
			continue
		}
		n := op.Weights.Rows * op.Weights.Cols
		need := n * 4
		if off+need > len(blob) {
			return fmt.Errorf("fountain: blob short")
		}
		w := make([]float32, n)
		for i := 0; i < n; i++ {
			w[i] = math.Float32frombits(binary.LittleEndian.Uint32(blob[off+i*4:]))
		}
		if err := op.Weights.SetFromF32(w); err != nil {
			return err
		}
		off += need
	}
	return nil
}

// Batch is one training example (float32 features + target).
type Batch struct {
	X, Y *core.Tensor[float32]
}

// Neural trains K dense specialists on shard batches, fountain-recovers weight blobs.
func Neural(factory Factory, batches []Batch, cfg Config) (*Master, error) {
	if factory == nil {
		return nil, fmt.Errorf("fountain: nil factory")
	}
	if cfg.K <= 0 {
		cfg = DefaultConfig()
	}
	blobs := make([][]byte, cfg.K)
	for i := 0; i < cfg.K; i++ {
		g, err := factory(i)
		if err != nil {
			return nil, err
		}
		// shard batches
		for e := 0; e < cfg.Epochs; e++ {
			for bi, b := range batches {
				if bi%cfg.K != i {
					continue
				}
				fwd, err := forward.Forward(g, b.X)
				if err != nil {
					return nil, err
				}
				if _, err := training.Step(fwd, b.Y, cfg.LR); err != nil {
					return nil, err
				}
			}
		}
		blob, err := PackGridWeights(g)
		if err != nil {
			return nil, err
		}
		blobs[i] = blob
	}
	// pad to equal size
	max := 0
	for _, b := range blobs {
		if len(b) > max {
			max = len(b)
		}
	}
	for i := range blobs {
		if len(blobs[i]) < max {
			pad := make([]byte, max)
			copy(pad, blobs[i])
			blobs[i] = pad
		}
	}
	rec, _, _, err := RecoverWeightBlobs(blobs, cfg.Seed, cfg.LossRate, cfg.MaxOverhead)
	if err != nil {
		return nil, err
	}
	dim := 0
	if g, err := factory(0); err == nil {
		if cell := g.At(0, 0, 0, 0); cell != nil {
			if op, ok := cell.Op.(*dense.Layer); ok {
				dim = op.Core.InputHeight
			}
		}
	}
	return &Master{Blobs: rec, Dim: dim, K: cfg.K}, nil
}
