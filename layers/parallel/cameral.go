package parallel

import (
	"fmt"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
)

// Hemispheres builds a Parallel of n Dense twins (same geometry), merged by combine.
// This is the cameral merge Op; nest it inside a Stack for multi-layer / nested cameral.
func Hemispheres(dim, outFeat, n int, combine CombineMode, act core.ActivationType, dt core.DType, format quant.Format) (*Layer, error) {
	if n < 1 {
		return nil, fmt.Errorf("parallel: Hemispheres needs n≥1")
	}
	if outFeat <= 0 {
		outFeat = dim
	}
	branches := make([]any, n)
	for i := 0; i < n; i++ {
		ch, err := dense.NewConfigured[float32](dim, outFeat, act, dt, format, nil)
		if err != nil {
			return nil, fmt.Errorf("hemisphere %d: %w", i, err)
		}
		branches[i] = ch
	}
	return NewFromBranches(Config{
		Dim: dim, OutFeat: outFeat, Branches: n, Combine: combine,
	}, branches, nil)
}

// HemispheresFrom is NewFromBranches sugar for explicit branch Ops (heterogeneous cameral).
// Branches may be any Parallel-supported Op (Dense, MHA, CNN, Sequential, …).
func HemispheresFrom(cfg Config, branches []any, gate *dense.Layer) (*Layer, error) {
	return NewFromBranches(cfg, branches, gate)
}

// CameralFromBranches wraps arbitrary hemisphere Ops in Stack[Parallel] — the
// poly-layer cameral entry point (any kind Parallel already accepts).
func CameralFromBranches(cfg Config, branches []any, gate *dense.Layer) (*Stack, error) {
	hemi, err := NewFromBranches(cfg, branches, gate)
	if err != nil {
		return nil, fmt.Errorf("cameral: %w", err)
	}
	return NewStack(hemi)
}

// BicameralFrom builds Stack[stem, Parallel(branches…), head] for mixed-type
// hemispheres when stem/head widths match combine output.
func BicameralFrom(stem any, cfg Config, branches []any, gate *dense.Layer, head any) (*Stack, error) {
	if stem == nil || head == nil {
		return nil, fmt.Errorf("parallel: BicameralFrom needs stem and head")
	}
	hemi, err := NewFromBranches(cfg, branches, gate)
	if err != nil {
		return nil, err
	}
	return NewStack(stem, hemi, head)
}

// Sandwich builds Stack(stem…, mid…, head…) — typically stem → cameral Parallel → head.
func Sandwich(ops ...any) (*Stack, error) {
	return NewStack(ops...)
}

// Bicameral builds Dense(in→hidden) → Parallel(2×Dense hidden→hidden, add) → Dense(hidden→out).
// Matches the test41 / tide bicameral sandwich.
func Bicameral(in, hidden, out int, act core.ActivationType, dt core.DType, format quant.Format) (*Stack, error) {
	if in <= 0 || hidden <= 0 || out <= 0 {
		return nil, fmt.Errorf("parallel: Bicameral needs positive in/hidden/out")
	}
	stem, err := dense.NewConfigured[float32](in, hidden, act, dt, format, nil)
	if err != nil {
		return nil, fmt.Errorf("bicameral stem: %w", err)
	}
	hemi, err := Hemispheres(hidden, hidden, 2, CombineAdd, act, dt, format)
	if err != nil {
		return nil, fmt.Errorf("bicameral hemi: %w", err)
	}
	head, err := dense.NewConfigured[float32](hidden, out, core.ActivationLinear, dt, format, nil)
	if err != nil {
		return nil, fmt.Errorf("bicameral head: %w", err)
	}
	return NewStack(stem, hemi, head)
}

// PlaceStack binds a Stack onto the volumetric grid at (z,y,x,l).
func PlaceStack(g *architecture.Grid, z, y, x, lidx int, layer *Stack) error {
	if g == nil || layer == nil {
		return fmt.Errorf("parallel: PlaceStack nil grid/layer")
	}
	layer.Core.Type = core.LayerStack
	layer.Core.Z, layer.Core.Y, layer.Core.X, layer.Core.L = z, y, x, lidx
	layer.Exec = g.Exec
	layer.SyncChildExec()
	return g.BindOp(z, y, x, lidx, layer.Core, layer)
}
