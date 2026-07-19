// Package tween implements neural target propagation (loom/poly tween rebuild).
//
// Default UseChainRule path: forward.Forward + layer Backward honor Exec.Backend
// (BackendSIMD → dense DotTile / Saxpy). Layerwise Hebbian updates use
// simd.SaxpyF32AccF64 for outer products; link budgets use simd.DotTile.
//
// Tests live in github.com/openfluke/w2a — not here.
package tween

import (
	"fmt"
	"math"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/systems/dna"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/simd"
)

// Config holds tunable parameters for neural target propagation.
type Config struct {
	BatchSize        int
	UseChainRule     bool
	GradientScale    float32
	DepthScaleFactor float32
	Momentum         float32
	LearningRate     float32
	ActivationClamp  float32
}

// TweenConfig is the loom-compatible alias.
type TweenConfig = Config

// DefaultConfig returns standard tween settings.
func DefaultConfig() *Config {
	return &Config{
		BatchSize:        1,
		UseChainRule:     true,
		GradientScale:    0.1,
		DepthScaleFactor: 1.1,
		Momentum:         0.9,
		LearningRate:     0.01,
		ActivationClamp:  10.0,
	}
}

// DefaultTweenConfig is the loom-compatible alias.
func DefaultTweenConfig() *Config { return DefaultConfig() }

// State tracks bidirectional signal flow for one network.
type State[T core.Numeric] struct {
	ForwardActs     []*core.Tensor[T]
	PreActs         []*core.Tensor[T]
	BackwardTargets []*core.Tensor[T]

	Gradients       []*core.Tensor[float32]
	WeightGradients []*core.Tensor[float32]
	WeightVel       [][]float32

	LinkBudgets []float32
	Gaps        []float32

	Config      *Config
	TotalLayers int
}

// NewState allocates tween buffers for a grid.
func NewState[T core.Numeric](g *architecture.Grid, config *Config) *State[T] {
	if config == nil {
		config = DefaultConfig()
	}
	total := 0
	if g != nil {
		total = g.StackLayerCount()
	}
	return &State[T]{
		ForwardActs:     make([]*core.Tensor[T], total+1),
		PreActs:         make([]*core.Tensor[T], total+1),
		BackwardTargets: make([]*core.Tensor[T], total+1),
		Gradients:       make([]*core.Tensor[float32], total+1),
		WeightGradients: make([]*core.Tensor[float32], total+1),
		WeightVel:       make([][]float32, total+1),
		LinkBudgets:     make([]float32, total),
		Gaps:            make([]float32, total),
		Config:          config,
		TotalLayers:     total,
	}
}

// NewTweenState is the loom-compatible alias.
func NewTweenState[T core.Numeric](g *architecture.Grid, config *Config) *State[T] {
	return NewState[T](g, config)
}

// CaptureFromForward fills ForwardActs/PreActs from a forward.Result tape.
func CaptureFromForward[T core.Numeric](s *State[T], fwd *forward.Result[T], input *core.Tensor[T]) {
	if s == nil || fwd == nil {
		return
	}
	s.ForwardActs[0] = input.Clone()
	for i, st := range fwd.Steps {
		idx := i + 1
		if idx >= len(s.ForwardActs) {
			break
		}
		s.PreActs[idx] = st.Pre
		s.ForwardActs[idx] = st.Post
	}
}

// Forward runs a volumetric forward and captures activations.
func Forward[T core.Numeric](g *architecture.Grid, s *State[T], input *core.Tensor[T]) (*core.Tensor[T], error) {
	if g == nil || s == nil || input == nil {
		return nil, fmt.Errorf("tween: nil grid/state/input")
	}
	fwd, err := forward.Forward(g, input)
	if err != nil {
		return nil, err
	}
	CaptureFromForward(s, fwd, input)
	return fwd.Output, nil
}

// TweenForward is the loom-compatible alias.
func TweenForward[T core.Numeric](g *architecture.Grid, s *State[T], input *core.Tensor[T]) (*core.Tensor[T], error) {
	return Forward(g, s, input)
}

// Backward generates targets from output back toward input.
func Backward[T core.Numeric](g *architecture.Grid, s *State[T], target *core.Tensor[T]) error {
	if s == nil || s.Config == nil {
		return fmt.Errorf("tween: nil state")
	}
	if s.Config.UseChainRule {
		return BackwardChainRule(g, s, target)
	}
	return BackwardLayerwise(g, s, target)
}

// TweenBackward is the loom-compatible alias.
func TweenBackward[T core.Numeric](g *architecture.Grid, s *State[T], target *core.Tensor[T]) error {
	return Backward(g, s, target)
}

// BackwardChainRule uses output error gradients to shift per-layer targets.
func BackwardChainRule[T core.Numeric](g *architecture.Grid, s *State[T], target *core.Tensor[T]) error {
	if s == nil || target == nil {
		return fmt.Errorf("tween: nil state/target")
	}
	outputIdx := s.TotalLayers
	actual := s.ForwardActs[outputIdx]
	for actual == nil && outputIdx > 0 {
		outputIdx--
		actual = s.ForwardActs[outputIdx]
	}
	if actual == nil {
		return fmt.Errorf("tween: no forward activations")
	}
	if actual.Len() != target.Len() {
		return fmt.Errorf("tween: target shape mismatch")
	}
	s.BackwardTargets[outputIdx] = target.Clone()

	grad := core.NewTensor[float32](target.Shape...)
	for i := range grad.Data {
		grad.Data[i] = float32(core.AsFloat64(target.Data[i]) - core.AsFloat64(actual.Data[i]))
	}
	s.Gradients[outputIdx] = grad

	order := hopCells(g)
	currentGrad := grad
	for i := len(order) - 1; i >= 0; i-- {
		cell := order[i]
		if cell == nil || cell.Layer.IsDisabled {
			s.Gradients[i] = currentGrad
			if i+1 < len(s.BackwardTargets) {
				s.BackwardTargets[i] = s.BackwardTargets[i+1]
			}
			continue
		}
		input := s.ForwardActs[i]
		preAct := s.PreActs[i+1]
		if preAct == nil {
			preAct = s.ForwardActs[i+1]
		}
		if input == nil {
			continue
		}

		gy := core.NewTensor[T](currentGrad.Shape...)
		for j := range gy.Data {
			gy.Data[j] = core.FromFloat64[T](float64(currentGrad.Data[j]))
		}
		gInT, gWT, err := layerBackwardAny(cell, gy, input, preAct)
		if err != nil {
			s.BackwardTargets[i] = input.Clone()
			s.Gradients[i] = currentGrad
			continue
		}
		gIn := core.NewTensor[float32](gInT.Shape...)
		for j := range gIn.Data {
			gIn.Data[j] = float32(core.AsFloat64(gInT.Data[j]))
		}
		s.Gradients[i] = gIn
		if gWT != nil {
			gW := core.NewTensor[float32](gWT.Shape...)
			for j := range gW.Data {
				gW.Data[j] = float32(core.AsFloat64(gWT.Data[j]))
			}
			s.WeightGradients[i] = gW
		}
		currentGrad = gIn

		targetT := core.NewTensor[T](input.Shape...)
		scale := s.Config.GradientScale
		for j := range targetT.Data {
			val := float32(core.AsFloat64(input.Data[j])) + gIn.Data[j]*scale
			targetT.Data[j] = core.FromFloat64[T](float64(val))
		}
		s.BackwardTargets[i] = targetT
	}
	return nil
}

// TweenBackwardChainRule is the loom-compatible alias.
func TweenBackwardChainRule[T core.Numeric](g *architecture.Grid, s *State[T], target *core.Tensor[T]) error {
	return BackwardChainRule(g, s, target)
}

// BackwardLayerwise uses true target propagation without derivatives (Dense).
func BackwardLayerwise[T core.Numeric](g *architecture.Grid, s *State[T], target *core.Tensor[T]) error {
	if s == nil || target == nil {
		return fmt.Errorf("tween: nil state/target")
	}
	outputIdx := s.TotalLayers
	for s.ForwardActs[outputIdx] == nil && outputIdx > 0 {
		outputIdx--
	}
	s.BackwardTargets[outputIdx] = target.Clone()

	order := hopCells(g)
	for i := len(order) - 1; i >= 0; i-- {
		cell := order[i]
		if cell == nil || cell.Layer.IsDisabled {
			if i+1 < len(s.BackwardTargets) {
				s.BackwardTargets[i] = s.BackwardTargets[i+1]
			}
			continue
		}
		currentTarget := s.BackwardTargets[i+1]
		input := s.ForwardActs[i]
		if currentTarget == nil || input == nil {
			continue
		}
		estimated := core.NewTensor[T](input.Shape...)
		if dl, ok := cell.Op.(*dense.Layer); ok && dl != nil && dl.Weights != nil {
			outSize := dl.Core.OutputHeight
			inSize := dl.Core.InputHeight
			w, err := dl.Weights.FlattenF32()
			if err == nil {
				tgt := make([]float32, outSize)
				for out := 0; out < outSize && out < currentTarget.Len(); out++ {
					tgt[out] = float32(core.AsFloat64(currentTarget.Data[out]))
				}
				imp, l1, err := hebbianImportances(w, tgt, outSize, inSize)
				if err == nil {
					for in := 0; in < inSize && in < estimated.Len(); in++ {
						if l1[in] > 0.01 {
							estimated.Data[in] = core.FromFloat64[T](float64(imp[in] / l1[in]))
						}
					}
				}
			}
		} else {
			copy(estimated.Data, input.Data)
		}
		s.BackwardTargets[i] = estimated
	}
	return nil
}

// TweenBackwardLayerwise is the loom-compatible alias.
func TweenBackwardLayerwise[T core.Numeric](g *architecture.Grid, s *State[T], target *core.Tensor[T]) error {
	return BackwardLayerwise(g, s, target)
}

// CalculateLinkBudgets measures cosine fidelity between forward acts and targets.
func (s *State[T]) CalculateLinkBudgets() {
	if s == nil {
		return
	}
	for i := 0; i < s.TotalLayers; i++ {
		fwd := s.ForwardActs[i+1]
		bwd := s.BackwardTargets[i+1]
		if fwd == nil || bwd == nil || fwd.Len() == 0 || fwd.Len() != bwd.Len() {
			continue
		}
		n := fwd.Len()
		a := make([]float32, n)
		b := make([]float32, n)
		for j := 0; j < n; j++ {
			a[j] = float32(core.AsFloat64(fwd.Data[j]))
			b[j] = float32(core.AsFloat64(bwd.Data[j]))
		}
		dot := simd.DotTile(a, b, 0, n, 0)
		fMag := simd.DotTile(a, a, 0, n, 0)
		bMag := simd.DotTile(b, b, 0, n, 0)
		gap := 0.0
		for j := 0; j < n; j++ {
			diff := float64(a[j] - b[j])
			gap += diff * diff
		}
		if fMag > 0 && bMag > 0 {
			cosine := dot / (math.Sqrt(fMag) * math.Sqrt(bMag))
			s.LinkBudgets[i] = float32((cosine + 1) / 2)
		}
		s.Gaps[i] = float32(math.Sqrt(gap / float64(n)))
	}
}

// ApplyGaps assigns weight updates from tween diagnostics.
func ApplyGaps[T core.Numeric](g *architecture.Grid, s *State[T], lr float32) error {
	if s == nil || s.Config == nil {
		return fmt.Errorf("tween: nil state")
	}
	if s.Config.UseChainRule {
		return applyChainRuleAll(g, s, lr)
	}
	return applyGapsLayerwiseAll(g, s, lr)
}

// ApplyTweenGaps is the loom-compatible alias.
func ApplyTweenGaps[T core.Numeric](g *architecture.Grid, s *State[T], lr float32) error {
	return ApplyGaps(g, s, lr)
}

func hopCells(g *architecture.Grid) []*architecture.Cell {
	if g == nil {
		return nil
	}
	order := g.HopOrder()
	out := make([]*architecture.Cell, len(order))
	for i, c := range order {
		out[i] = g.At(c.Z, c.Y, c.X, c.L)
	}
	return out
}

// WeightF32 reads one flattened weight (test helper).
func WeightF32(op any, idx int) (float32, error) {
	stores := dna.CollectStores(op)
	off := 0
	for _, s := range stores {
		if s == nil {
			continue
		}
		w, err := s.FlattenF32()
		if err != nil {
			return 0, err
		}
		if idx < off+len(w) {
			return w[idx-off], nil
		}
		off += len(w)
	}
	return 0, fmt.Errorf("tween: weight index %d out of range", idx)
}

// TweenWeightF32 is the loom-compatible alias.
func TweenWeightF32(op any, idx int) (float32, error) { return WeightF32(op, idx) }
