package universal

import (
	"fmt"
	"math"
	"sort"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/model/hf"
	"github.com/openfluke/welvet/quant"
)

// TensorMeta holds geometric and statistical metadata for a tensor.
type TensorMeta struct {
	Idx           int
	Name          string
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

// LoadUniversal loads a model from a safetensors file: probe geometry, then
// mount it (Dense/MHA/SwiGLU archetypes get real, weight-loaded Ops; anything
// else falls back to a Dense placeholder of matching in/out).
func LoadUniversal(path string) (*architecture.Grid, error) {
	_, archs, _, geoms, err := LoadUniversalDetailed(path)
	if err != nil {
		return nil, err
	}
	g := MountGeometrically(archs, geoms)
	if g == nil {
		return nil, fmt.Errorf("universal: %q mounted an empty grid", path)
	}
	return g, nil
}

// LoadUniversalDetailed reads every 2-D/1-D tensor from a safetensors file,
// probes deep geometry, and returns (tensorCount, archetypes, missedIndices, geoms).
// Tensor order is sorted by name for determinism (safetensors headers are a JSON
// object with no guaranteed key order).
func LoadUniversalDetailed(path string) (int, []LayerArchetype, []int, []TensorMeta, error) {
	names, err := hf.TensorNames(path)
	if err != nil {
		return 0, nil, nil, nil, fmt.Errorf("universal: %q: %w", path, err)
	}
	sort.Strings(names)
	tensors, err := hf.LoadSafetensorsWithMeta(path, nil)
	if err != nil {
		return 0, nil, nil, nil, fmt.Errorf("universal: %q: %w", path, err)
	}
	geoms := make([]TensorMeta, 0, len(names))
	for _, name := range names {
		info, ok := tensors[name]
		if !ok {
			continue
		}
		meanAbs, variance := WeightDistribution(info.Data)
		geoms = append(geoms, TensorMeta{
			Idx:           len(geoms),
			Name:          name,
			Shape:         info.Shape,
			Data:          info.Data,
			MeanAbs:       meanAbs,
			Variance:      variance,
			Rank:          len(info.Shape),
			OriginalDType: core.DTypeFloat32,
		})
	}
	archs, missed := ProbeDeepGeometry(geoms)
	return len(geoms), archs, missed, geoms, nil
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

// MountGeometrically creates a Grid from archetypes. Dense/MHA/SwiGLU archetypes
// mount real weight-loaded Ops (from the source tensor Data); anything else (or
// any mount failure) falls back to a zero-weight Dense placeholder of matching
// in/out so the grid still has the right layer count and shape.
func MountGeometrically(archs []LayerArchetype, geoms []TensorMeta) *architecture.Grid {
	n := len(archs)
	if n < 1 {
		n = 1
	}
	g := architecture.NewGrid(1, 1, 1, n)
	for i, a := range archs {
		if mountArchetype(g, i, a, geoms) {
			continue
		}
		in, out := denseFallbackDims(a)
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

func denseFallbackDims(a LayerArchetype) (in, out int) {
	in, out = 8, 8
	if a.GeomMetrics != nil {
		if v, ok := a.GeomMetrics["in"]; ok {
			in = v
		}
		if v, ok := a.GeomMetrics["d"]; ok {
			in, out = v, v
		}
		if v, ok := a.GeomMetrics["out"]; ok {
			out = v
		}
	}
	return in, out
}

// mountArchetype places a real Op for known types; returns false (no Op placed)
// on any shape/data mismatch so the caller can fall back to a Dense placeholder.
func mountArchetype(g *architecture.Grid, lidx int, a LayerArchetype, geoms []TensorMeta) bool {
	switch a.Type {
	case core.LayerDense:
		wIdx, ok := a.Indices["w"]
		if !ok || wIdx >= len(geoms) {
			return false
		}
		gm := geoms[wIdx]
		if gm.Rank != 2 || len(gm.Data) != gm.Shape[0]*gm.Shape[1] {
			return false
		}
		l, err := dense.NewConfigured(gm.Shape[1], gm.Shape[0], core.ActivationLinear, gm.OriginalDType, quant.FormatNone, gm.Data)
		if err != nil {
			return false
		}
		return dense.Place(g, 0, 0, 0, lidx, l) == nil

	case core.LayerMultiHeadAttention:
		qIdx, qHas := a.Indices["q"]
		kIdx, kHas := a.Indices["k"]
		vIdx, vHas := a.Indices["v"]
		oIdx, oHas := a.Indices["o"]
		if !qHas || !kHas || !vHas || !oHas {
			return false
		}
		qW, qOK := denseData(geoms, qIdx)
		kW, kOK := denseData(geoms, kIdx)
		vW, vOK := denseData(geoms, vIdx)
		oW, oOK := denseData(geoms, oIdx)
		if !qOK || !kOK || !vOK || !oOK {
			return false
		}
		dim := a.GeomMetrics["d"]
		if dim <= 0 {
			return false
		}
		heads := 1
		if dim%64 == 0 && dim > 64 {
			heads = dim / 64 // heuristic: 64-wide heads are the common convention
		}
		cfg := mha.Config{DModel: dim, NumHeads: heads, HeadDim: dim / heads}
		l, err := mha.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone, qW, kW, vW, oW)
		if err != nil {
			return false
		}
		return mha.Place(g, 0, 0, 0, lidx, l) == nil

	case core.LayerSwiGLU:
		gIdx, gHas := a.Indices["g"]
		uIdx, uHas := a.Indices["u"]
		dIdx, dHas := a.Indices["d"]
		if !gHas || !uHas || !dHas {
			return false
		}
		gW, gOK := denseData(geoms, gIdx)
		uW, uOK := denseData(geoms, uIdx)
		dW, dOK := denseData(geoms, dIdx)
		gGeom, gGOK := geomAt(geoms, gIdx)
		dGeom, dGOK := geomAt(geoms, dIdx)
		if !gOK || !uOK || !dOK || !gGOK || !dGOK {
			return false
		}
		inter := gGeom.Shape[0]
		inDim := dGeom.Shape[0]
		if inter <= 0 || inDim <= 0 {
			return false
		}
		cfg := swiglu.Config{InputDim: inDim, IntermediateDim: inter}
		l, err := swiglu.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone, gW, uW, dW)
		if err != nil {
			return false
		}
		return swiglu.Place(g, 0, 0, 0, lidx, l) == nil

	default:
		return false
	}
}

func geomAt(geoms []TensorMeta, idx int) (TensorMeta, bool) {
	if idx < 0 || idx >= len(geoms) {
		return TensorMeta{}, false
	}
	return geoms[idx], true
}

func denseData(geoms []TensorMeta, idx int) ([]float32, bool) {
	gm, ok := geomAt(geoms, idx)
	if !ok || gm.Rank != 2 || len(gm.Data) != gm.Shape[0]*gm.Shape[1] {
		return nil, false
	}
	return gm.Data, true
}
