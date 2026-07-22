package dna

import (
	"fmt"
	"math"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/cnn2"
	"github.com/openfluke/welvet/layers/cnn3"
	"github.com/openfluke/welvet/layers/convt1"
	"github.com/openfluke/welvet/layers/convt2"
	"github.com/openfluke/welvet/layers/convt3"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/embedding"
	"github.com/openfluke/welvet/layers/gdn"
	"github.com/openfluke/welvet/layers/kmeans"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/lstm"
	"github.com/openfluke/welvet/layers/mamba"
	"github.com/openfluke/welvet/layers/metacognition"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/rnn"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/layers/softmax"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/weights"
)

// LayerSignature is the unique 3D topological "DNA" of one grid cell.
type LayerSignature struct {
	Z, Y, X, L int
	Type       core.LayerType
	DType      core.DType
	Weights    []float32 // L2-normalized, scale-applied weights
}

// NetworkDNA is the complete genetic blueprint of a Grid.
type NetworkDNA []LayerSignature

// ExtractDNA generates topological signatures for all cells in hop order.
func ExtractDNA(g *architecture.Grid) NetworkDNA {
	if g == nil {
		return nil
	}
	order := g.HopOrder()
	out := make(NetworkDNA, 0, len(order))
	for _, c := range order {
		cell := g.At(c.Z, c.Y, c.X, c.L)
		if cell == nil {
			continue
		}
		out = append(out, LayerSignature{
			Z: c.Z, Y: c.Y, X: c.X, L: c.L,
			Type:    cell.Layer.Type,
			DType:   cell.Layer.DType,
			Weights: signatureFromCell(cell),
		})
	}
	return out
}

func signatureFromCell(cell *architecture.Cell) []float32 {
	if cell == nil {
		return []float32{1.0}
	}
	switch cell.Layer.Type {
	case core.LayerSoftmax:
		return []float32{1.0}
	case core.LayerParallel:
		var flat []float32
		for i := range cell.ParallelBranches {
			b := &cell.ParallelBranches[i]
			if b.IsRemoteLink {
				continue
			}
			flat = append(flat, signatureFromCell(b)...)
		}
		if len(flat) == 0 {
			return []float32{1.0}
		}
		return Normalize(flat)
	case core.LayerSequential:
		if op, ok := cell.Op.(*sequential.Layer); ok && op != nil {
			var flat []float32
			for _, ch := range op.Children {
				flat = append(flat, flattenStores(storesFromDense(ch))...)
			}
			if len(flat) == 0 {
				return []float32{1.0}
			}
			return Normalize(flat)
		}
		var flat []float32
		for i := range cell.SequentialLayers {
			flat = append(flat, signatureFromCell(&cell.SequentialLayers[i])...)
		}
		if len(flat) == 0 {
			return []float32{1.0}
		}
		return Normalize(flat)
	case core.LayerResidual:
		if op, ok := cell.Op.(*residual.Layer); ok && op != nil {
			var flat []float32
			for _, ch := range op.Children {
				flat = append(flat, flattenStores(storesFromDense(ch))...)
			}
			if len(flat) == 0 {
				return []float32{1.0}
			}
			return Normalize(flat)
		}
		return []float32{1.0}
	default:
		flat, err := FlattenOp(cell.Op)
		if err != nil || len(flat) == 0 {
			return []float32{1.0}
		}
		return Normalize(flat)
	}
}

func storesFromDense(d *dense.Layer) []*weights.Store {
	if d == nil || d.Weights == nil {
		return nil
	}
	return []*weights.Store{d.Weights}
}

func flattenStores(stores []*weights.Store) []float32 {
	var flat []float32
	for _, s := range stores {
		if s == nil {
			continue
		}
		v, err := s.FlattenF32()
		if err != nil || len(v) == 0 {
			continue
		}
		flat = append(flat, v...)
	}
	return flat
}

// CollectStores returns every *weights.Store owned by a cell Op.
func CollectStores(op any) []*weights.Store {
	if op == nil {
		return nil
	}
	switch v := op.(type) {
	case *dense.Layer:
		return storesFromDense(v)
	case *mha.Layer:
		var out []*weights.Store
		out = append(out, storesFromDense(v.Q)...)
		out = append(out, storesFromDense(v.K)...)
		out = append(out, storesFromDense(v.V)...)
		out = append(out, storesFromDense(v.O)...)
		return out
	case *swiglu.Layer:
		var out []*weights.Store
		out = append(out, storesFromDense(v.Gate)...)
		out = append(out, storesFromDense(v.Up)...)
		out = append(out, storesFromDense(v.Down)...)
		return out
	case *rmsnorm.Layer:
		if v.Gamma != nil {
			return []*weights.Store{v.Gamma}
		}
	case *layernorm.Layer:
		var out []*weights.Store
		if v.Gamma != nil {
			out = append(out, v.Gamma)
		}
		if v.Beta != nil {
			out = append(out, v.Beta)
		}
		return out
	case *cnn1.Layer:
		return storesFromDense(v.Proj)
	case *cnn2.Layer:
		return storesFromDense(v.Proj)
	case *cnn3.Layer:
		return storesFromDense(v.Proj)
	case *convt1.Layer:
		return storesFromDense(v.Proj)
	case *convt2.Layer:
		return storesFromDense(v.Proj)
	case *convt3.Layer:
		return storesFromDense(v.Proj)
	case *rnn.Layer:
		var out []*weights.Store
		out = append(out, storesFromDense(v.IH)...)
		out = append(out, storesFromDense(v.HH)...)
		return out
	case *lstm.Layer:
		return lstmStores(v)
	case *embedding.Layer:
		if v.Weights != nil {
			return []*weights.Store{v.Weights}
		}
	case *sequential.Layer:
		var out []*weights.Store
		for _, ch := range v.Children {
			out = append(out, storesFromDense(ch)...)
		}
		return out
	case *residual.Layer:
		var out []*weights.Store
		for _, ch := range v.Children {
			out = append(out, storesFromDense(ch)...)
		}
		return out
	case *parallel.Layer:
		var out []*weights.Store
		for _, ch := range v.Branches {
			out = append(out, CollectStores(ch)...)
		}
		out = append(out, storesFromDense(v.Gate)...)
		return out
	case *kmeans.Layer:
		return storesFromDense(v.Centers)
	case *mamba.Layer:
		var out []*weights.Store
		out = append(out, storesFromDense(v.InProj)...)
		out = append(out, storesFromDense(v.OutProj)...)
		return out
	case *metacognition.Layer:
		return storesFromDense(v.Observed)
	case *softmax.Layer:
		return nil
	case *gdn.Layer:
		// GDN holds quant.Blob projections — no weights.Store; DNA uses FlattenOp.
		return nil
	}
	return nil
}

// FlattenOp returns concatenated scale-applied float32 weights for any Op
// (including GDN blobs). Empty → nil (caller may use neutral marker).
func FlattenOp(op any) ([]float32, error) {
	var flat []float32
	for _, s := range CollectStores(op) {
		if s == nil {
			continue
		}
		v, err := s.FlattenF32()
		if err != nil {
			return nil, err
		}
		flat = append(flat, v...)
	}
	if gl, ok := op.(*gdn.Layer); ok && gl != nil {
		for _, b := range []*quant.Blob{gl.InQKV, gl.InZ, gl.InB, gl.InA, gl.Out} {
			if b == nil {
				continue
			}
			u, err := quant.Unpack(b)
			if err != nil {
				return nil, err
			}
			flat = append(flat, u...)
		}
		flat = append(flat, gl.ConvWeight...)
		flat = append(flat, gl.ALog...)
		flat = append(flat, gl.DtBias...)
		flat = append(flat, gl.NormGamma...)
	}
	return flat, nil
}

func lstmStores(l *lstm.Layer) []*weights.Store {
	if l == nil {
		return nil
	}
	var out []*weights.Store
	for _, g := range []*lstm.Gate{l.I, l.F, l.G, l.O} {
		if g == nil {
			continue
		}
		out = append(out, storesFromDense(g.IH)...)
		out = append(out, storesFromDense(g.HH)...)
	}
	return out
}

// Normalize returns the L2 unit vector of v (zeros stay zeros).
func Normalize(v []float32) []float32 {
	var sumSq float64
	for _, x := range v {
		sumSq += float64(x) * float64(x)
	}
	mag := float32(math.Sqrt(sumSq))
	if mag == 0 {
		return make([]float32, len(v))
	}
	res := make([]float32, len(v))
	for i, x := range v {
		res[i] = x / mag
	}
	return res
}

// CosineSimilarity compares two signatures (−1…1). Architectural mismatch → 0.
func CosineSimilarity(s1, s2 LayerSignature) float32 {
	if s1.Type != s2.Type || s1.DType != s2.DType {
		return 0
	}
	if len(s1.Weights) != len(s2.Weights) {
		return 0
	}
	isZ1, isZ2 := true, true
	var dot float32
	for i := range s1.Weights {
		if s1.Weights[i] != 0 {
			isZ1 = false
		}
		if s2.Weights[i] != 0 {
			isZ2 = false
		}
		dot += s1.Weights[i] * s2.Weights[i]
	}
	if isZ1 && isZ2 {
		return 1.0
	}
	if isZ1 || isZ2 {
		return 0.0
	}
	return dot
}

// NetworkComparisonResult holds hierarchical similarity metrics.
type NetworkComparisonResult struct {
	OverallOverlap float32
	LayerOverlaps  map[string]float32
	LogicShifts    []LogicShift
}

// LogicShift identifies a pattern that moved in (Z,Y,X,L) space.
type LogicShift struct {
	SourcePos string
	TargetPos string
	Overlap   float32
}

// CompareNetworks performs hierarchical spatial correlation between two blueprints.
func CompareNetworks(dna1, dna2 NetworkDNA) NetworkComparisonResult {
	res := NetworkComparisonResult{
		LayerOverlaps: make(map[string]float32),
		LogicShifts:   []LogicShift{},
	}
	var totalOverlap float32
	var matchedCount int

	for _, sig1 := range dna1 {
		posKey := fmt.Sprintf("%d,%d,%d,%d", sig1.Z, sig1.Y, sig1.X, sig1.L)
		for _, sig2 := range dna2 {
			if sig1.Z == sig2.Z && sig1.Y == sig2.Y && sig1.X == sig2.X && sig1.L == sig2.L {
				overlap := CosineSimilarity(sig1, sig2)
				res.LayerOverlaps[posKey] = overlap
				totalOverlap += overlap
				matchedCount++
				break
			}
		}
	}
	if matchedCount > 0 {
		res.OverallOverlap = totalOverlap / float32(matchedCount)
	}

	for _, sig1 := range dna1 {
		bestOverlap := float32(-1.0)
		bestSig2 := LayerSignature{}
		found := false
		for _, sig2 := range dna2 {
			if len(sig1.Weights) != len(sig2.Weights) {
				continue
			}
			overlap := CosineSimilarity(sig1, sig2)
			if overlap > bestOverlap {
				bestOverlap = overlap
				bestSig2 = sig2
				found = true
			}
		}
		if found && bestOverlap > 0.8 {
			pos1 := fmt.Sprintf("%d,%d,%d,%d", sig1.Z, sig1.Y, sig1.X, sig1.L)
			pos2 := fmt.Sprintf("%d,%d,%d,%d", bestSig2.Z, bestSig2.Y, bestSig2.X, bestSig2.L)
			if pos1 != pos2 {
				res.LogicShifts = append(res.LogicShifts, LogicShift{
					SourcePos: pos1,
					TargetPos: pos2,
					Overlap:   bestOverlap,
				})
			}
		}
	}
	return res
}
