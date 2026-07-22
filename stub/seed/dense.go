package seed

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/fnv"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/cnn2"
	"github.com/openfluke/welvet/layers/cnn3"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/quant"
)

const denseManifestFormat = "welvet-dense-weight-v1"

// DenseLayerManifest is one Dense layer seed recipe.
type DenseLayerManifest struct {
	Index     int    `json:"index"`
	Path      string `json:"path"`
	In        int    `json:"in"`
	Out       int    `json:"out"`
	LayerSeed uint64 `json:"layer_seed"`
	DType     string `json:"dtype"`
	WeightFP  uint64 `json:"weight_fp"`
}

// DenseManifest is a topology seed + per-layer He recipes (no weight bytes).
type DenseManifest struct {
	Format       string               `json:"format"`
	TopologySeed uint64               `json:"topology_seed"`
	Sizes        []int                `json:"sizes"`
	Layers       []DenseLayerManifest `json:"layers"`
	NetworkFP    uint64               `json:"network_fp"`
}

// DenseLayerWeightSeed derives the seed for dense stack layer i.
func DenseLayerWeightSeed(topologySeed uint64, i int) uint64 {
	return DeriveLayer(topologySeed, i, fmt.Sprintf("dense[%d]", i))
}

// BuildDense creates a DenseManifest and He-inits fingerprints.
func BuildDense(topologySeed uint64, sizes []int, dtypes []string) (*DenseManifest, error) {
	if len(sizes) < 2 {
		return nil, fmt.Errorf("seed: need ≥2 sizes")
	}
	m := &DenseManifest{
		Format:       denseManifestFormat,
		TopologySeed: topologySeed,
		Sizes:        append([]int(nil), sizes...),
		Layers:       make([]DenseLayerManifest, 0, len(sizes)-1),
	}
	h := fnv.New64a()
	var buf [8]byte
	for i := 0; i < len(sizes)-1; i++ {
		in, out := sizes[i], sizes[i+1]
		dt := "Float32"
		if i < len(dtypes) && dtypes[i] != "" {
			dt = dtypes[i]
		}
		seed := DenseLayerWeightSeed(topologySeed, i)
		w := make([]float32, in*out)
		InitFloat32He(w, in, seed)
		fp := FingerprintF32(w)
		binary.LittleEndian.PutUint64(buf[:], fp)
		_, _ = h.Write(buf[:])
		m.Layers = append(m.Layers, DenseLayerManifest{
			Index: i, Path: fmt.Sprintf("dense[%d]", i), In: in, Out: out,
			LayerSeed: seed, DType: dt, WeightFP: fp,
		})
	}
	m.NetworkFP = h.Sum64()
	return m, nil
}

// BuildDenseGrid materializes a 1×1×1×N grid of Dense layers from the manifest.
func BuildDenseGrid(m *DenseManifest) (*architecture.Grid, error) {
	if m == nil || len(m.Layers) == 0 {
		return nil, fmt.Errorf("seed: empty dense manifest")
	}
	n := len(m.Layers)
	g := architecture.NewGrid(1, 1, 1, n)
	for i, lm := range m.Layers {
		w := make([]float32, lm.In*lm.Out)
		InitFloat32He(w, lm.In, lm.LayerSeed)
		l, err := dense.NewConfigured(lm.In, lm.Out, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, w)
		if err != nil {
			return nil, err
		}
		if err := dense.Place(g, 0, 0, 0, i, l); err != nil {
			return nil, err
		}
	}
	return g, nil
}

// MarshalDense JSON-encodes a dense manifest.
func MarshalDense(m *DenseManifest) ([]byte, error) {
	return json.MarshalIndent(m, "", "  ")
}

// ParseDense decodes a dense manifest.
func ParseDense(data []byte) (*DenseManifest, error) {
	var m DenseManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

// InitGrid He-inits every Dense Op on the grid from initSeed.
func InitGrid(g *architecture.Grid, initSeed uint64) error {
	if g == nil {
		return fmt.Errorf("seed: nil grid")
	}
	idx := 0
	for _, c := range g.HopOrder() {
		cell := g.At(c.Z, c.Y, c.X, c.L)
		if cell == nil || cell.Op == nil {
			continue
		}
		path := fmt.Sprintf("cell[%d,%d,%d,%d]", c.Z, c.Y, c.X, c.L)
		layerSeed := DeriveLayer(initSeed, idx, path)
		idx++
		switch op := cell.Op.(type) {
		case *dense.Layer:
			if err := initDenseHe(op, layerSeed); err != nil {
				return err
			}
		case *mha.Layer:
			for _, part := range []struct {
				name string
				l    *dense.Layer
			}{{"q", op.Q}, {"k", op.K}, {"v", op.V}, {"o", op.O}} {
				if err := initDenseHe(part.l, DeriveLayer(layerSeed, 0, "mha."+part.name)); err != nil {
					return err
				}
			}
		case *swiglu.Layer:
			for _, part := range []struct {
				name string
				l    *dense.Layer
			}{{"gate", op.Gate}, {"up", op.Up}, {"down", op.Down}} {
				if err := initDenseHe(part.l, DeriveLayer(layerSeed, 0, "swiglu."+part.name)); err != nil {
					return err
				}
			}
		case *cnn1.Layer:
			if err := initDenseHe(op.Proj, DeriveLayer(layerSeed, 0, "cnn1.proj")); err != nil {
				return err
			}
		case *cnn2.Layer:
			if err := initDenseHe(op.Proj, DeriveLayer(layerSeed, 0, "cnn2.proj")); err != nil {
				return err
			}
		case *cnn3.Layer:
			if err := initDenseHe(op.Proj, DeriveLayer(layerSeed, 0, "cnn3.proj")); err != nil {
				return err
			}
		case *parallel.Layer:
			if err := initParallelHe(op, layerSeed); err != nil {
				return err
			}
		case *sequential.Layer:
			for ci, child := range op.Children {
				if err := initDenseHe(child, DeriveLayer(layerSeed, ci, "sequential.child")); err != nil {
					return err
				}
			}
		case *residual.Layer:
			for ci, child := range op.Children {
				if err := initDenseHe(child, DeriveLayer(layerSeed, ci, "residual.child")); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// initDenseHe He-inits one Dense Store, deriving InputHeight from Core or Cols.
func initDenseHe(op *dense.Layer, layerSeed uint64) error {
	if op == nil || op.Weights == nil {
		return nil
	}
	in := op.Core.InputHeight
	if in <= 0 {
		in = op.Weights.Cols
	}
	return InitStoreHe(op.Weights, in, layerSeed)
}

func initParallelHe(op *parallel.Layer, layerSeed uint64) error {
	if op == nil {
		return nil
	}
	for bi, br := range op.Branches {
		bs := DeriveLayer(layerSeed, bi, "parallel.branch")
		switch b := br.(type) {
		case *dense.Layer:
			if err := initDenseHe(b, bs); err != nil {
				return err
			}
		case *parallel.Layer:
			if err := initParallelHe(b, bs); err != nil {
				return err
			}
		case *mha.Layer:
			for _, part := range []struct {
				name string
				l    *dense.Layer
			}{{"q", b.Q}, {"k", b.K}, {"v", b.V}, {"o", b.O}} {
				if err := initDenseHe(part.l, DeriveLayer(bs, 0, "mha."+part.name)); err != nil {
					return err
				}
			}
		case *swiglu.Layer:
			for _, part := range []struct {
				name string
				l    *dense.Layer
			}{{"gate", b.Gate}, {"up", b.Up}, {"down", b.Down}} {
				if err := initDenseHe(part.l, DeriveLayer(bs, 0, "swiglu."+part.name)); err != nil {
					return err
				}
			}
		case *sequential.Layer:
			for ci, child := range b.Children {
				if err := initDenseHe(child, DeriveLayer(bs, ci, "sequential.child")); err != nil {
					return err
				}
			}
		case *residual.Layer:
			for ci, child := range b.Children {
				if err := initDenseHe(child, DeriveLayer(bs, ci, "residual.child")); err != nil {
					return err
				}
			}
		case *cnn1.Layer:
			if err := initDenseHe(b.Proj, DeriveLayer(bs, 0, "cnn1.proj")); err != nil {
				return err
			}
		case *cnn2.Layer:
			if err := initDenseHe(b.Proj, DeriveLayer(bs, 0, "cnn2.proj")); err != nil {
				return err
			}
		case *cnn3.Layer:
			if err := initDenseHe(b.Proj, DeriveLayer(bs, 0, "cnn3.proj")); err != nil {
				return err
			}
		}
	}
	if op.Gate != nil {
		if err := initDenseHe(op.Gate, DeriveLayer(layerSeed, 0, "parallel.gate")); err != nil {
			return err
		}
	}
	return nil
}

func fingerprintParallel(op *parallel.Layer, write func(*dense.Layer)) {
	if op == nil {
		return
	}
	for _, br := range op.Branches {
		switch b := br.(type) {
		case *dense.Layer:
			write(b)
		case *parallel.Layer:
			fingerprintParallel(b, write)
		case *mha.Layer:
			write(b.Q)
			write(b.K)
			write(b.V)
			write(b.O)
		case *swiglu.Layer:
			write(b.Gate)
			write(b.Up)
			write(b.Down)
		case *sequential.Layer:
			for _, child := range b.Children {
				write(child)
			}
		case *residual.Layer:
			for _, child := range b.Children {
				write(child)
			}
		case *cnn1.Layer:
			write(b.Proj)
		case *cnn2.Layer:
			write(b.Proj)
		case *cnn3.Layer:
			write(b.Proj)
		}
	}
	write(op.Gate)
}

// GridFingerprint hashes all Dense (direct or nested in mha/swiglu/cnnN/parallel/
// sequential/residual) FlattenF32 weights in hop order.
func GridFingerprint(g *architecture.Grid) uint64 {
	if g == nil {
		return 0
	}
	h := fnv.New64a()
	var buf [8]byte
	write := func(op *dense.Layer) {
		if op == nil {
			return
		}
		fp := StoreFingerprint(op.Weights)
		binary.LittleEndian.PutUint64(buf[:], fp)
		_, _ = h.Write(buf[:])
	}
	for _, c := range g.HopOrder() {
		cell := g.At(c.Z, c.Y, c.X, c.L)
		if cell == nil {
			continue
		}
		switch op := cell.Op.(type) {
		case *dense.Layer:
			write(op)
		case *mha.Layer:
			write(op.Q)
			write(op.K)
			write(op.V)
			write(op.O)
		case *swiglu.Layer:
			write(op.Gate)
			write(op.Up)
			write(op.Down)
		case *cnn1.Layer:
			write(op.Proj)
		case *cnn2.Layer:
			write(op.Proj)
		case *cnn3.Layer:
			write(op.Proj)
		case *parallel.Layer:
			fingerprintParallel(op, write)
		case *sequential.Layer:
			for _, child := range op.Children {
				write(child)
			}
		case *residual.Layer:
			for _, child := range op.Children {
				write(child)
			}
		}
	}
	return h.Sum64()
}
