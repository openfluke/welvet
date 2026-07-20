package universal

import (
	"fmt"
	"math"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
)

// TensorMeta holds geometric and statistical metadata for a tensor.
type TensorMeta struct {
	Idx           int
	Shape         []int
	Data          []float32
	MeanAbs       float32
	Variance      float32
	Rank          int
	OriginalDType core.DType
}

// LayerArchetype represents a detected structural unit in the model.
type LayerArchetype struct {
	Type        core.LayerType
	TypeName    string
	Indices     map[string]int
	GeomMetrics map[string]int
}

// UserHints allows manual mapping for ambiguous tensor indices.
var UserHints = make(map[int]core.LayerType)

// LoadUniversal loads a model from safetensors (not wired in welvet v0).
func LoadUniversal(path string) (*architecture.Grid, error) {
	_, _, _, _, err := LoadUniversalDetailed(path)
	if err != nil {
		return nil, err
	}
	return nil, fmt.Errorf("universal: safetensors not wired (path %q)", path)
}

// LoadUniversalDetailed performs deep analysis — requires external tensor supply in v0.
func LoadUniversalDetailed(path string) (int, []LayerArchetype, []int, []TensorMeta, error) {
	return 0, nil, nil, nil, fmt.Errorf("universal: safetensors not wired (path %q)", path)
}

// WeightDistribution computes mean absolute value and variance.
func WeightDistribution(data []float32) (meanAbs, variance float32) {
	if len(data) == 0 {
		return 0, 0
	}
	var sumAbs, sumSq float64
	for _, v := range data {
		av := math.Abs(float64(v))
		sumAbs += av
		sumSq += av * av
	}
	mean := float32(sumAbs / float64(len(data)))
	varSq := float32(sumSq/float64(len(data)) - float64(mean*mean))
	return mean, varSq
}

// ProbeDeepGeometry identifies layer patterns within a set of tensors.
func ProbeDeepGeometry(geoms []TensorMeta) ([]LayerArchetype, []int) {
	var archetypes []LayerArchetype
	used := make(map[int]bool)

	for idx, hType := range UserHints {
		if idx < len(geoms) {
			used[idx] = true
			archetypes = append(archetypes, LayerArchetype{
				Type: hType, TypeName: "HINTED Layer",
				Indices: map[string]int{"w": idx},
			})
		}
	}

	for i := range geoms {
		if used[i] {
			continue
		}
		if arch, ok := matchMHA(geoms, i, used); ok {
			archetypes = append(archetypes, arch)
			continue
		}
		if arch, ok := matchFFN(geoms, i, used); ok {
			archetypes = append(archetypes, arch)
			continue
		}
	}

	for i := range geoms {
		if used[i] {
			continue
		}
		g := geoms[i]
		if g.Rank == 2 {
			used[i] = true
			arch := LayerArchetype{Indices: map[string]int{"w": i}, GeomMetrics: map[string]int{"out": g.Shape[0], "in": g.Shape[1]}}
			if g.Shape[0] > g.Shape[1]*10 {
				arch.Type = core.LayerEmbedding
				arch.TypeName = "Embedding"
				arch.GeomMetrics = map[string]int{"v": g.Shape[0], "d": g.Shape[1]}
			} else {
				arch.Type = core.LayerDense
				arch.TypeName = "Dense Linear"
			}
			archetypes = append(archetypes, arch)
		} else if g.Rank == 1 && g.MeanAbs > 0.4 {
			used[i] = true
			archetypes = append(archetypes, LayerArchetype{
				Type: core.LayerRMSNorm, TypeName: "Normalization",
				Indices: map[string]int{"w": i}, GeomMetrics: map[string]int{"d": g.Shape[0]},
			})
		}
	}

	var missed []int
	for i := range geoms {
		if !used[i] {
			missed = append(missed, i)
		}
	}
	return archetypes, missed
}

func matchMHA(geoms []TensorMeta, pivot int, used map[int]bool) (LayerArchetype, bool) {
	g := geoms[pivot]
	if g.Rank != 2 || g.Shape[0] != g.Shape[1] {
		return LayerArchetype{}, false
	}
	dim := g.Shape[0]
	cluster := []int{pivot}
	for j, o := range geoms {
		if used[j] || j == pivot || o.Rank != 2 || o.Shape[0] != dim || o.Shape[1] != dim {
			continue
		}
		cluster = append(cluster, j)
		if len(cluster) == 4 {
			break
		}
	}
	if len(cluster) != 4 {
		return LayerArchetype{}, false
	}
	for _, idx := range cluster {
		used[idx] = true
	}
	return LayerArchetype{
		Type: core.LayerMultiHeadAttention, TypeName: "MHA",
		Indices:     map[string]int{"q": cluster[0], "k": cluster[1], "v": cluster[2], "o": cluster[3]},
		GeomMetrics: map[string]int{"d": dim},
	}, true
}

func matchFFN(geoms []TensorMeta, pivot int, used map[int]bool) (LayerArchetype, bool) {
	g := geoms[pivot]
	if g.Rank != 2 {
		return LayerArchetype{}, false
	}
	da, db := g.Shape[0], g.Shape[1]
	cluster := []int{pivot}
	for j, o := range geoms {
		if used[j] || j == pivot || o.Rank != 2 {
			continue
		}
		if (o.Shape[0] == da && o.Shape[1] == db) || (o.Shape[0] == db && o.Shape[1] == da) {
			cluster = append(cluster, j)
			if len(cluster) == 3 {
				break
			}
		}
	}
	if len(cluster) != 3 {
		return LayerArchetype{}, false
	}
	for _, idx := range cluster {
		used[idx] = true
	}
	return LayerArchetype{Type: core.LayerSwiGLU, TypeName: "SwiGLU", Indices: map[string]int{"g": cluster[0], "u": cluster[1], "d": cluster[2]}}, true
}

// MountGeometrically creates a Grid of Dense placeholders from archetypes.
func MountGeometrically(archs []LayerArchetype, geoms []TensorMeta) *architecture.Grid {
	n := len(archs)
	if n < 1 {
		n = 1
	}
	g := architecture.NewGrid(1, 1, 1, n)
	for i, a := range archs {
		in, out := 8, 8
		if a.GeomMetrics != nil {
			if v, ok := a.GeomMetrics["in"]; ok {
				in = v
			}
			if v, ok := a.GeomMetrics["d"]; ok {
				in = v
				out = v
			}
			if v, ok := a.GeomMetrics["out"]; ok {
				out = v
			}
		}
		dt := core.DTypeFloat32
		if wIdx, ok := a.Indices["w"]; ok && wIdx < len(geoms) {
			dt = geoms[wIdx].OriginalDType
		}
		l, _ := dense.New(in, out, core.ActivationLinear, dt)
		if l != nil {
			_ = dense.Place(g, 0, 0, 0, i, l)
		}
	}
	return g
}
