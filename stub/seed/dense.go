package seed

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/fnv"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
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
			in := op.Core.InputHeight
			if in <= 0 {
				in = op.Weights.Cols
			}
			if err := InitStoreHe(op.Weights, in, layerSeed); err != nil {
				return err
			}
		}
	}
	return nil
}

// GridFingerprint hashes all Dense FlattenF32 weights in hop order.
func GridFingerprint(g *architecture.Grid) uint64 {
	if g == nil {
		return 0
	}
	h := fnv.New64a()
	var buf [8]byte
	for _, c := range g.HopOrder() {
		cell := g.At(c.Z, c.Y, c.X, c.L)
		if cell == nil {
			continue
		}
		if op, ok := cell.Op.(*dense.Layer); ok {
			fp := StoreFingerprint(op.Weights)
			binary.LittleEndian.PutUint64(buf[:], fp)
			_, _ = h.Write(buf[:])
		}
	}
	return h.Sum64()
}
