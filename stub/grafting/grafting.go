package grafting

import (
	"fmt"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/quant"
)

// GraftGrids collects the first cell Op from each grid and builds one Parallel layer.
func GraftGrids(grids []*architecture.Grid, combine parallel.CombineMode) (*parallel.Layer, error) {
	if len(grids) == 0 {
		return nil, fmt.Errorf("grafting: no grids")
	}
	var branches []*dense.Layer
	dim, outFeat := 0, 0
	for _, g := range grids {
		if g == nil {
			continue
		}
		cell := g.GetLayer(0, 0, 0, 0)
		if cell == nil || cell.Op == nil {
			continue
		}
		switch op := cell.Op.(type) {
		case *parallel.Layer:
			for _, b := range op.Branches {
				if b == nil {
					continue
				}
				branches = append(branches, b)
				if dim == 0 {
					dim = b.Core.InputHeight
					outFeat = b.Core.OutputHeight
				}
			}
		case *dense.Layer:
			branches = append(branches, op)
			if dim == 0 {
				dim = op.Core.InputHeight
				outFeat = op.Core.OutputHeight
			}
		default:
			return nil, fmt.Errorf("grafting: unsupported Op %T at (0,0,0,0) — v0 Dense only", cell.Op)
		}
	}
	if len(branches) == 0 {
		return nil, fmt.Errorf("grafting: no valid Dense branches")
	}
	if dim <= 0 || outFeat <= 0 {
		return nil, fmt.Errorf("grafting: invalid branch geometry")
	}
	cfg := parallel.Config{
		Dim: dim, OutFeat: outFeat, Branches: len(branches), Combine: combine,
	}
	l, err := parallel.NewConfigured[float32](cfg, core.DTypeFloat32, quant.FormatNone, nil, nil)
	if err != nil {
		return nil, err
	}
	for i, b := range branches {
		l.Branches[i] = b
	}
	return l, nil
}

// ResidualGraft wraps the first cell Dense in a Residual block (y = F(x) + x).
func ResidualGraft(g *architecture.Grid) (*residual.Layer, error) {
	if g == nil {
		return nil, fmt.Errorf("grafting: nil grid")
	}
	cell := g.GetLayer(0, 0, 0, 0)
	if cell == nil || cell.Op == nil {
		return nil, fmt.Errorf("grafting: empty cell (0,0,0,0)")
	}
	dl, ok := cell.Op.(*dense.Layer)
	if !ok {
		return nil, fmt.Errorf("grafting: ResidualGraft needs *dense.Layer at (0,0,0,0); use GraftGrids+CombineAdd for other ops")
	}
	dim := dl.Core.InputHeight
	if dim <= 0 {
		dim = dl.Core.OutputHeight
	}
	cfg := residual.Config{Dim: dim, Depth: 1}
	l, err := residual.NewConfigured[float32](cfg, core.DTypeFloat32, quant.FormatNone, nil)
	if err != nil {
		return nil, err
	}
	l.Children[0] = dl
	return l, nil
}

// GraftGridsHeterogeneous is the loom-parity alias requested for GraftGrids —
// same Dense/Parallel → Parallel behavior, kept as a distinct name because the
// loom source separates the "same-family" combinator from the fully generic
// GraftToGrid below.
func GraftGridsHeterogeneous(grids []*architecture.Grid, combine parallel.CombineMode) (*parallel.Layer, error) {
	return GraftGrids(grids, combine)
}

// GraftToGrid places each source grid's cell(0,0,0,0) Op into a new
// 1×1×1×N grid — a sequential stack, one Op per L, for ANY Op type
// (dense/mha/swiglu/cnnN/mamba/gdn/parallel/…). It copies the source cell's
// already-populated core.Layer descriptor + Op pointer via Grid.BindOp, so it
// needs no per-layer-package import or type switch, unlike GraftGrids (which
// is Dense/Parallel-only because it builds one *parallel.Layer with typed
// []*dense.Layer branches).
//
// This is a sequential (not additive) stack — combine happens by chaining
// forward through L, not by summing branch outputs. Use ResidualGraft for a
// true y = F(x) + x skip connection (Dense only — see ResidualGraftGrid).
func GraftToGrid(grids []*architecture.Grid) (*architecture.Grid, error) {
	if len(grids) == 0 {
		return nil, fmt.Errorf("grafting: no grids")
	}
	var cells []*architecture.Cell
	for _, g := range grids {
		if g == nil {
			continue
		}
		cell := g.GetLayer(0, 0, 0, 0)
		if cell == nil || cell.Op == nil {
			continue
		}
		cells = append(cells, cell)
	}
	if len(cells) == 0 {
		return nil, fmt.Errorf("grafting: no valid Ops at (0,0,0,0)")
	}
	out := architecture.NewGrid(1, 1, 1, len(cells))
	for i, cell := range cells {
		meta := cell.Layer
		meta.Z, meta.Y, meta.X, meta.L = 0, 0, 0, i
		if err := out.BindOp(0, 0, 0, i, meta, cell.Op); err != nil {
			return nil, fmt.Errorf("grafting: GraftToGrid bind %d (%T): %w", i, cell.Op, err)
		}
	}
	return out, nil
}

// ResidualGraftGrid generalizes ResidualGraft to a *architecture.Grid return type
// (uniform with GraftToGrid). For *dense.Layer it delegates to ResidualGraft
// (a real y = F(x) + x skip connection). For any other Op it returns a clear
// error: Cell.ParallelBranches / Cell.CombineMode exist on architecture.Cell but
// are not read by runtime/forward or runtime/backward (verified — dead fields),
// so an additive skip connection over a non-Dense Op cannot actually execute
// today. Callers that just need the Op to run (without the +x skip) should use
// GraftToGrid instead.
func ResidualGraftGrid(g *architecture.Grid) (*architecture.Grid, error) {
	if g == nil {
		return nil, fmt.Errorf("grafting: nil grid")
	}
	cell := g.GetLayer(0, 0, 0, 0)
	if cell == nil || cell.Op == nil {
		return nil, fmt.Errorf("grafting: empty cell (0,0,0,0)")
	}
	if _, ok := cell.Op.(*dense.Layer); ok {
		rl, err := ResidualGraft(g)
		if err != nil {
			return nil, err
		}
		out := architecture.NewGrid(1, 1, 1, 1)
		if err := residual.Place(out, 0, 0, 0, 0, rl); err != nil {
			return nil, err
		}
		return out, nil
	}
	return nil, fmt.Errorf(
		"grafting: ResidualGraftGrid needs *dense.Layer at (0,0,0,0), got %T — "+
			"additive skip connections for non-Dense Ops need Cell.ParallelBranches/"+
			"CombineMode wired into runtime/forward+backward first (currently unread "+
			"by both); use GraftToGrid for a sequential (non-additive) stack instead",
		cell.Op)
}
