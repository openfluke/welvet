// Package evolution extends DNA with splice crossover and NEAT-style mutation
// (loom/poly evolution rebuild). Operates on architecture.Grid; children are
// CPU-resident clones (no GPU device state).
//
// Tests live in github.com/openfluke/w2a — not here.
package evolution

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/dna"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/embedding"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/softmax"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/quant"
)

// SpliceConfig controls how two parent networks are combined.
type SpliceConfig struct {
	CrossoverMode string  // "uniform", "point", or "blend"
	BlendAlpha    float32 // 0=all A, 1=all B
	SplitRatio    float64
	FitnessA      float64
	FitnessB      float64
}

// DefaultSpliceConfig returns a balanced blend configuration.
func DefaultSpliceConfig() SpliceConfig {
	return SpliceConfig{
		CrossoverMode: "blend",
		BlendAlpha:    0.5,
		SplitRatio:    0.5,
	}
}

// SpliceResult holds the outcome of a DNA splice operation.
type SpliceResult struct {
	Child        *architecture.Grid
	ParentADNA   dna.NetworkDNA
	ParentBDNA   dna.NetworkDNA
	ChildDNA     dna.NetworkDNA
	Similarities map[string]float32
	BlendedCount int
}

// SpliceDNA merges two trained parents into a child (structural template = parentA).
func SpliceDNA(parentA, parentB *architecture.Grid, cfg SpliceConfig) (*architecture.Grid, error) {
	res, err := SpliceDNAWithReport(parentA, parentB, cfg)
	if err != nil {
		return nil, err
	}
	return res.Child, nil
}

// SpliceDNAWithReport performs a splice and returns diagnostics.
func SpliceDNAWithReport(parentA, parentB *architecture.Grid, cfg SpliceConfig) (SpliceResult, error) {
	var empty SpliceResult
	if parentA == nil || parentB == nil {
		return empty, fmt.Errorf("evolution: nil parent")
	}
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	dnaA := dna.ExtractDNA(parentA)
	dnaB := dna.ExtractDNA(parentB)

	bSigMap := make(map[string]dna.LayerSignature, len(dnaB))
	for _, sig := range dnaB {
		bSigMap[layerKey(sig.Z, sig.Y, sig.X, sig.L)] = sig
	}
	aSigMap := make(map[string]dna.LayerSignature, len(dnaA))
	for _, sig := range dnaA {
		aSigMap[layerKey(sig.Z, sig.Y, sig.X, sig.L)] = sig
	}

	sims := make(map[string]float32)
	blended := 0
	for _, sigA := range dnaA {
		key := layerKey(sigA.Z, sigA.Y, sigA.X, sigA.L)
		if sigB, ok := bSigMap[key]; ok {
			sim := dna.CosineSimilarity(sigA, sigB)
			sims[key] = sim
			if sim > 0 {
				blended++
			}
		}
	}

	child, err := CloneGrid(parentA)
	if err != nil {
		return empty, err
	}

	for _, coord := range parentA.HopOrder() {
		cellA := parentA.At(coord.Z, coord.Y, coord.X, coord.L)
		cellB := parentB.At(coord.Z, coord.Y, coord.X, coord.L)
		cellC := child.At(coord.Z, coord.Y, coord.X, coord.L)
		if cellA == nil || cellB == nil || cellC == nil {
			continue
		}
		storesA := dna.CollectStores(cellA.Op)
		storesB := dna.CollectStores(cellB.Op)
		storesC := dna.CollectStores(cellC.Op)
		if len(storesA) == 0 || len(storesA) != len(storesB) || len(storesA) != len(storesC) {
			continue
		}

		key := layerKey(coord.Z, coord.Y, coord.X, coord.L)
		sigA := aSigMap[key]
		sigB := bSigMap[key]
		similarity := dna.CosineSimilarity(sigA, sigB)

		for i := range storesA {
			wA, err := storesA[i].FlattenF32()
			if err != nil {
				continue
			}
			wB, err := storesB[i].FlattenF32()
			if err != nil || len(wB) != len(wA) {
				continue
			}
			childW := make([]float32, len(wA))
			switch cfg.CrossoverMode {
			case "point":
				split := int(float64(len(wA)) * cfg.SplitRatio)
				if split > len(wA) {
					split = len(wA)
				}
				if split < 0 {
					split = 0
				}
				copy(childW[:split], wA[:split])
				copy(childW[split:], wB[split:])
			case "uniform":
				threshold := float32(0.5)
				if cfg.FitnessA+cfg.FitnessB > 0 {
					threshold = float32(cfg.FitnessA / (cfg.FitnessA + cfg.FitnessB))
				}
				for j := range childW {
					if rng.Float32() < threshold {
						childW[j] = wA[j]
					} else {
						childW[j] = wB[j]
					}
				}
			default: // blend
				alpha := cfg.BlendAlpha
				if cfg.FitnessA+cfg.FitnessB > 0 {
					fitnessAlpha := float32(cfg.FitnessB / (cfg.FitnessA + cfg.FitnessB))
					alpha = fitnessAlpha * (0.5 + 0.5*similarity)
				}
				for j := range childW {
					childW[j] = wA[j]*(1-alpha) + wB[j]*alpha
				}
			}
			if err := storesC[i].SetFromF32(childW); err != nil {
				return empty, fmt.Errorf("evolution splice %s store %d: %w", key, i, err)
			}
		}
	}

	return SpliceResult{
		Child:        child,
		ParentADNA:   dnaA,
		ParentBDNA:   dnaB,
		ChildDNA:     dna.ExtractDNA(child),
		Similarities: sims,
		BlendedCount: blended,
	}, nil
}

// NEATConfig controls mutation probabilities.
type NEATConfig struct {
	WeightPerturbRate  float64
	WeightPerturbScale float32
	NodeMutateRate     float64
	ConnectionAddRate  float64
	ConnectionDropRate float64
	ActivationMutRate  float64
	LayerToggleRate    float64

	AllowedLayerTypes []core.LayerType
	DModel            int
	Seed              int64
}

// DefaultNEATConfig returns conservative mutation rates.
func DefaultNEATConfig(dModel int) NEATConfig {
	return NEATConfig{
		WeightPerturbRate:  0.8,
		WeightPerturbScale: 0.05,
		NodeMutateRate:     0.1,
		ConnectionAddRate:  0.05,
		ConnectionDropRate: 0.02,
		ActivationMutRate:  0.1,
		LayerToggleRate:    0.02,
		DModel:             dModel,
		AllowedLayerTypes: []core.LayerType{
			core.LayerDense, core.LayerSoftmax, core.LayerRMSNorm, core.LayerLayerNorm,
			core.LayerSwiGLU, core.LayerMultiHeadAttention, core.LayerEmbedding,
			core.LayerResidual, core.LayerSequential,
		},
		Seed: time.Now().UnixNano(),
	}
}

// NEATMutate applies NEAT-style mutations to a clone of g.
func NEATMutate(g *architecture.Grid, cfg NEATConfig) (*architecture.Grid, error) {
	if g == nil {
		return nil, fmt.Errorf("evolution: nil grid")
	}
	child, err := CloneGrid(g)
	if err != nil {
		return nil, err
	}
	rng := rand.New(rand.NewSource(cfg.Seed))
	if cfg.DModel <= 0 {
		cfg.DModel = 8
	}

	for _, coord := range child.HopOrder() {
		cell := child.At(coord.Z, coord.Y, coord.X, coord.L)
		if cell == nil {
			continue
		}
		if rng.Float64() < cfg.WeightPerturbRate {
			for _, s := range dna.CollectStores(cell.Op) {
				if s == nil {
					continue
				}
				w, err := s.FlattenF32()
				if err != nil {
					continue
				}
				perturbWeights(w, cfg.WeightPerturbScale, rng)
				_ = s.SetFromF32(w)
			}
		}
		if rng.Float64() < cfg.ActivationMutRate {
			act := randomActivation(rng)
			cell.Layer.Activation = act
			setOpActivation(cell.Op, act)
		}
		if rng.Float64() < cfg.NodeMutateRate {
			newType := randomLayerType(cfg.AllowedLayerTypes, cell.Layer.Type, rng)
			if newType != cell.Layer.Type {
				if err := reinitCell(cell, newType, cfg, rng); err != nil {
					return nil, err
				}
			}
		}
		if rng.Float64() < cfg.LayerToggleRate {
			cell.Layer.IsDisabled = !cell.Layer.IsDisabled
		}
	}

	if rng.Float64() < cfg.ConnectionAddRate {
		addConnection(child, rng)
	}
	if rng.Float64() < cfg.ConnectionDropRate {
		dropConnection(child, rng)
	}
	return child, nil
}

// NEATPopulation manages a pool of networks evolving over generations.
type NEATPopulation struct {
	Networks  []*architecture.Grid
	Fitnesses []float64
	Config    NEATConfig
	rng       *rand.Rand
}

// NewNEATPopulation creates an initial population by mutating a seed network.
func NewNEATPopulation(seed *architecture.Grid, size int, cfg NEATConfig) (*NEATPopulation, error) {
	if seed == nil || size < 1 {
		return nil, fmt.Errorf("evolution: bad population args")
	}
	pop := &NEATPopulation{
		Networks:  make([]*architecture.Grid, size),
		Fitnesses: make([]float64, size),
		Config:    cfg,
		rng:       rand.New(rand.NewSource(cfg.Seed)),
	}
	for i := range pop.Networks {
		mutCfg := cfg
		mutCfg.Seed = pop.rng.Int63()
		n, err := NEATMutate(seed, mutCfg)
		if err != nil {
			return nil, err
		}
		pop.Networks[i] = n
	}
	return pop, nil
}

// Evolve runs one generation given a fitness function (higher = better).
func (p *NEATPopulation) Evolve(fitnessFn func(*architecture.Grid) float64) error {
	if p == nil || fitnessFn == nil {
		return fmt.Errorf("evolution: nil population/fitness")
	}
	for i, net := range p.Networks {
		p.Fitnesses[i] = fitnessFn(net)
	}
	p.sortByFitness()

	size := len(p.Networks)
	eliteCount := size / 4
	if eliteCount < 1 {
		eliteCount = 1
	}
	next := make([]*architecture.Grid, size)
	for i := 0; i < eliteCount; i++ {
		next[i] = p.Networks[i]
	}
	for i := eliteCount; i < size; i++ {
		aIdx := p.rng.Intn(eliteCount)
		bIdx := p.rng.Intn(eliteCount)
		for bIdx == aIdx && eliteCount > 1 {
			bIdx = p.rng.Intn(eliteCount)
		}
		spliceCfg := DefaultSpliceConfig()
		spliceCfg.FitnessA = p.Fitnesses[aIdx]
		spliceCfg.FitnessB = p.Fitnesses[bIdx]
		child, err := SpliceDNA(p.Networks[aIdx], p.Networks[bIdx], spliceCfg)
		if err != nil {
			return err
		}
		mutCfg := p.Config
		mutCfg.Seed = p.rng.Int63()
		mut, err := NEATMutate(child, mutCfg)
		if err != nil {
			return err
		}
		next[i] = mut
	}
	p.Networks = next
	return nil
}

// Best returns the highest-fitness network from the last Evolve call.
func (p *NEATPopulation) Best() *architecture.Grid {
	if p == nil || len(p.Networks) == 0 {
		return nil
	}
	return p.Networks[0]
}

// BestFitness returns the fitness score of the top network.
func (p *NEATPopulation) BestFitness() float64 {
	if p == nil || len(p.Fitnesses) == 0 {
		return 0
	}
	return p.Fitnesses[0]
}

// Summary prints a one-line diagnostic for the population.
func (p *NEATPopulation) Summary(generation int) string {
	if p == nil || len(p.Fitnesses) == 0 {
		return fmt.Sprintf("Gen %d: empty population", generation)
	}
	best := p.Fitnesses[0]
	worst := p.Fitnesses[len(p.Fitnesses)-1]
	var sum float64
	for _, f := range p.Fitnesses {
		sum += f
	}
	avg := sum / float64(len(p.Fitnesses))
	return fmt.Sprintf("Gen %d | best=%.4f  avg=%.4f  worst=%.4f  pop=%d",
		generation, best, avg, worst, len(p.Networks))
}

func (p *NEATPopulation) sortByFitness() {
	n := len(p.Networks)
	for i := 0; i < n-1; i++ {
		for j := 0; j < n-1-i; j++ {
			if p.Fitnesses[j] < p.Fitnesses[j+1] {
				p.Networks[j], p.Networks[j+1] = p.Networks[j+1], p.Networks[j]
				p.Fitnesses[j], p.Fitnesses[j+1] = p.Fitnesses[j+1], p.Fitnesses[j]
			}
		}
	}
}

func layerKey(z, y, x, l int) string {
	return fmt.Sprintf("%d,%d,%d,%d", z, y, x, l)
}

// CloneGrid deep-copies topology + Ops (CPU-resident weight stores).
func CloneGrid(src *architecture.Grid) (*architecture.Grid, error) {
	if src == nil {
		return nil, fmt.Errorf("evolution: nil grid")
	}
	dst := architecture.NewGrid(src.Depth, src.Rows, src.Cols, src.LayersPerCell)
	dst.Exec = src.Exec
	dst.NativeExact = src.NativeExact
	dst.Tanhi = src.Tanhi
	for i := range src.Cells {
		sc := src.Cells[i]
		dc := &dst.Cells[i]
		dc.Layer = sc.Layer
		dc.IsRemoteLink = sc.IsRemoteLink
		dc.TargetZ, dc.TargetY, dc.TargetX, dc.TargetL = sc.TargetZ, sc.TargetY, sc.TargetX, sc.TargetL
		dc.CombineMode = sc.CombineMode
		if len(sc.ParallelBranches) > 0 {
			dc.ParallelBranches = append([]architecture.Cell(nil), sc.ParallelBranches...)
		}
		if len(sc.SequentialLayers) > 0 {
			dc.SequentialLayers = append([]architecture.Cell(nil), sc.SequentialLayers...)
		}
		op, err := cloneOp(sc.Op)
		if err != nil {
			return nil, fmt.Errorf("evolution clone cell %d: %w", i, err)
		}
		dc.Op = op
	}
	return dst, nil
}

func perturbWeights(w []float32, scale float32, rng *rand.Rand) {
	for i := range w {
		w[i] += (rng.Float32()*2 - 1) * scale
	}
}

func randomActivation(rng *rand.Rand) core.ActivationType {
	acts := []core.ActivationType{
		core.ActivationReLU, core.ActivationSilu, core.ActivationGELU,
		core.ActivationTanh, core.ActivationSigmoid, core.ActivationLinear,
	}
	return acts[rng.Intn(len(acts))]
}

func randomLayerType(allowed []core.LayerType, current core.LayerType, rng *rand.Rand) core.LayerType {
	candidates := make([]core.LayerType, 0, len(allowed))
	for _, t := range allowed {
		if t != current {
			candidates = append(candidates, t)
		}
	}
	if len(candidates) == 0 {
		return current
	}
	return candidates[rng.Intn(len(candidates))]
}

func setOpActivation(op any, act core.ActivationType) {
	switch v := op.(type) {
	case *dense.Layer:
		if v != nil {
			v.Core.Activation = act
		}
	}
}

func randInit(n int, rng *rand.Rand, scale float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = (rng.Float32()*2 - 1) * scale
	}
	return out
}

func reinitCell(cell *architecture.Cell, newType core.LayerType, cfg NEATConfig, rng *rand.Rand) error {
	dModel := cfg.DModel
	if dModel < 1 {
		dModel = 8
	}
	cell.Layer.Type = newType
	cell.Layer.InputHeight = dModel
	cell.Layer.OutputHeight = dModel
	dt := core.DTypeFloat32
	format := quant.FormatNone
	switch newType {
	case core.LayerSoftmax:
		sm, err := softmax.New(softmax.Config{Dim: dModel, SeqLen: 1})
		if err != nil {
			return err
		}
		cell.Op = sm
		return nil
	case core.LayerDense:
		dl, err := dense.NewConfigured(dModel, dModel, cell.Layer.Activation, dt, format, randInit(dModel*dModel, rng, 0.05))
		if err != nil {
			return err
		}
		cell.Op = dl
		return nil
	case core.LayerRMSNorm:
		ones := make([]float32, dModel)
		for i := range ones {
			ones[i] = 1
		}
		rl, err := rmsnorm.NewConfigured(rmsnorm.Config{Dim: dModel}, dt, format, ones)
		if err != nil {
			return err
		}
		cell.Op = rl
		return nil
	case core.LayerLayerNorm:
		ones := make([]float32, dModel)
		zeros := make([]float32, dModel)
		for i := range ones {
			ones[i] = 1
		}
		ll, err := layernorm.NewConfigured(layernorm.Config{Dim: dModel}, dt, format, ones, zeros)
		if err != nil {
			return err
		}
		cell.Op = ll
		return nil
	case core.LayerSwiGLU:
		inter := dModel * 2
		sl, err := swiglu.NewConfigured(swiglu.Config{InputDim: dModel, IntermediateDim: inter}, dt, format,
			randInit(dModel*inter, rng, 0.05), randInit(dModel*inter, rng, 0.05), randInit(inter*dModel, rng, 0.05))
		if err != nil {
			return err
		}
		cell.Op = sl
		return nil
	case core.LayerMultiHeadAttention:
		heads := 4
		if dModel < 4 {
			heads = 1
		}
		for dModel%heads != 0 && heads > 1 {
			heads--
		}
		mcfg := mha.Config{DModel: dModel, NumHeads: heads, MaxSeqLen: 8}
		ml, err := mha.NewConfigured[float32](mcfg, dt, format, nil, nil, nil, nil)
		if err != nil {
			return err
		}
		cell.Op = ml
		return nil
	case core.LayerEmbedding:
		el, err := embedding.NewConfigured(embedding.Config{VocabSize: 64, EmbeddingDim: dModel, SeqLen: 4}, dt, format, randInit(64*dModel, rng, 0.02))
		if err != nil {
			return err
		}
		cell.Op = el
		return nil
	default:
		dl, err := dense.NewConfigured(dModel, dModel, cell.Layer.Activation, dt, format, randInit(dModel*dModel, rng, 0.05))
		if err != nil {
			return err
		}
		cell.Layer.Type = core.LayerDense
		cell.Op = dl
		return nil
	}
}

func addConnection(g *architecture.Grid, rng *rand.Rand) {
	order := g.HopOrder()
	if len(order) < 2 {
		return
	}
	a := order[rng.Intn(len(order))]
	b := order[rng.Intn(len(order))]
	for b == a {
		b = order[rng.Intn(len(order))]
	}
	_ = g.SetRemoteLink(a.Z, a.Y, a.X, a.L, b.Z, b.Y, b.X, b.L)
}

func dropConnection(g *architecture.Grid, rng *rand.Rand) {
	var candidates []architecture.Coord
	for _, c := range g.HopOrder() {
		cell := g.At(c.Z, c.Y, c.X, c.L)
		if cell != nil && cell.IsRemoteLink {
			candidates = append(candidates, c)
		}
	}
	if len(candidates) == 0 {
		return
	}
	pick := candidates[rng.Intn(len(candidates))]
	g.ClearRemoteLink(pick.Z, pick.Y, pick.X, pick.L)
}