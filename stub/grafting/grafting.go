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
