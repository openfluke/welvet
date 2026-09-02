package parallel

import (
	"time"

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
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/rnn"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/layers/softmax"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/systems/tanhi"
)

func init() {
	tanhi.RegisterTopologyBridge(bridgeCellTopology)
	tanhi.RegisterGridTanhiSync(syncGridTanhi)
}

// SyncGridTanhi copies grid.Tanhi into every Parallel / Stack host on the grid.
func SyncGridTanhi(g *architecture.Grid) {
	if g == nil {
		return
	}
	for i := range g.Cells {
		cell := &g.Cells[i]
		if cell.Op == nil {
			continue
		}
		syncTanhiIntoOp(cell.Op, g.Tanhi)
		stampCellFromOp(cell)
	}
}

func syncGridTanhi(g *architecture.Grid) { SyncGridTanhi(g) }

func syncTanhiIntoOp(op any, tanhiCfg any) {
	switch v := op.(type) {
	case *Layer:
		if v == nil {
			return
		}
		v.Tanhi = tanhiCfg
		for _, b := range v.Branches {
			syncTanhiIntoOp(b, tanhiCfg)
		}
	case *Stack:
		if v == nil {
			return
		}
		v.Tanhi = tanhiCfg
		for _, ch := range v.Children {
			syncTanhiIntoOp(ch, tanhiCfg)
		}
	case *sequential.Layer:
		if v == nil {
			return
		}
		v.Tanhi = tanhiCfg
		for _, ch := range v.ChildOps() {
			syncTanhiIntoOp(ch, tanhiCfg)
		}
	case *residual.Layer:
		if v == nil {
			return
		}
		for _, ch := range v.ChildOps() {
			syncTanhiIntoOp(ch, tanhiCfg)
		}
	}
}

func bridgeCellTopology(cell *architecture.Cell) {
	if cell == nil || cell.Op == nil {
		return
	}
	stampCellFromOp(cell)
}

func stampCellFromOp(cell *architecture.Cell) {
	switch cell.Layer.Type {
	case core.LayerParallel:
		if l, ok := cell.Op.(*Layer); ok && l != nil {
			stampParallelTopology(cell, l)
		}
	case core.LayerStack:
		if s, ok := cell.Op.(*Stack); ok && s != nil {
			stampStackTopology(cell, s)
		}
	case core.LayerSequential:
		if sl, ok := cell.Op.(*sequential.Layer); ok && sl != nil {
			stampSequentialTopology(cell, sl)
		}
	}
}

func stampParallelTopology(cell *architecture.Cell, l *Layer) {
	if cell == nil || l == nil {
		return
	}
	cell.CombineMode = string(l.Cfg.Combine)
	z, y, x, parentL := cell.Layer.Z, cell.Layer.Y, cell.Layer.X, cell.Layer.L
	cell.ParallelBranches = branchCells(l.Branches, z, y, x, parentL)
}

func stampStackTopology(cell *architecture.Cell, s *Stack) {
	if cell == nil || s == nil {
		return
	}
	z, y, x := cell.Layer.Z, cell.Layer.Y, cell.Layer.X
	cell.SequentialLayers = childCells(s.Children, z, y, x, cell.Layer.L)
}

func stampSequentialTopology(cell *architecture.Cell, l *sequential.Layer) {
	if cell == nil || l == nil {
		return
	}
	z, y, x := cell.Layer.Z, cell.Layer.Y, cell.Layer.X
	cell.SequentialLayers = childCells(l.ChildOps(), z, y, x, cell.Layer.L)
}

func branchCells(branches []any, z, y, x, parentL int) []architecture.Cell {
	out := make([]architecture.Cell, len(branches))
	for i, b := range branches {
		out[i] = childCell(b, z, y, x, parentL, i)
	}
	return out
}

func childCells(children []any, z, y, x, parentL int) []architecture.Cell {
	out := make([]architecture.Cell, len(children))
	for i, ch := range children {
		out[i] = childCell(ch, z, y, x, parentL, i)
	}
	return out
}

func childCell(op any, z, y, x, parentL, slot int) architecture.Cell {
	meta := opLayerMeta(op)
	meta.Z, meta.Y, meta.X = z, y, x
	meta.L = parentL
	_ = slot
	c := architecture.Cell{Layer: meta, Op: op}
	stampCellFromOp(&c)
	return c
}

func opLayerMeta(op any) core.Layer {
	if v, ok := op.(*View); ok && v != nil {
		return core.Layer{Type: core.LayerDense, DType: core.DTypeFloat32}
	}
	if _, ok := op.(*Flatten); ok {
		return core.Layer{Type: core.LayerDense, DType: core.DTypeFloat32}
	}
	switch v := op.(type) {
	case *dense.Layer:
		return v.Core
	case *mha.Layer:
		return v.Core
	case *swiglu.Layer:
		return v.Core
	case *rmsnorm.Layer:
		return v.Core
	case *layernorm.Layer:
		return v.Core
	case *softmax.Layer:
		return v.Core
	case *cnn1.Layer:
		return v.Core
	case *cnn2.Layer:
		return v.Core
	case *cnn3.Layer:
		return v.Core
	case *convt1.Layer:
		return v.Core
	case *convt2.Layer:
		return v.Core
	case *convt3.Layer:
		return v.Core
	case *rnn.Layer:
		return v.Core
	case *lstm.Layer:
		return v.Core
	case *embedding.Layer:
		return v.Core
	case *sequential.Layer:
		return v.Core
	case *residual.Layer:
		return v.Core
	case *Layer:
		return v.Core
	case *Stack:
		return v.Core
	case *kmeans.Layer:
		return v.Core
	case *mamba.Layer:
		return v.Core
	case *metacognition.Layer:
		return v.Core
	case *gdn.Layer:
		return core.Layer{Type: core.LayerGDN, DType: core.DTypeFloat32}
	case *ResidualSkip:
		return opLayerMeta(v.F)
	default:
		return core.Layer{Type: core.LayerDense, DType: core.DTypeFloat32}
	}
}

// SetTanhi attaches HUD config and propagates into nested branch Ops.
func (l *Layer) SetTanhi(cfg any) {
	if l == nil {
		return
	}
	l.Tanhi = cfg
	syncTanhiIntoOp(l, cfg)
}

// SetTanhi attaches HUD config and propagates into nested child Ops.
func (s *Stack) SetTanhi(cfg any) {
	if s == nil {
		return
	}
	s.Tanhi = cfg
	syncTanhiIntoOp(s, cfg)
}

func tanhiCfg(v any) *tanhi.UDPConfig {
	return tanhi.ConfigFromAny(v)
}

func shapeFromTensorT[T core.Numeric](cfg *tanhi.UDPConfig, t *core.Tensor[T]) []int {
	if cfg == nil || !cfg.SendShape || t == nil {
		return nil
	}
	return append([]int(nil), t.Shape...)
}

func emitOpTanhi[T core.Numeric](cfgAny any, phase string, idx int, op any, t0, t1 time.Time, act *core.Tensor[T]) {
	cfg := tanhiCfg(cfgAny)
	if cfg == nil {
		return
	}
	meta := opLayerMeta(op)
	cell := architecture.Cell{Layer: meta, Op: op}
	stampCellFromOp(&cell)
	tanhi.Emit(cfg, phase, idx, &cell, t0, t1, shapeFromTensorT(cfg, act))
}

func emitLayerTanhi[T core.Numeric](l *Layer, phase string, idx int, t0, t1 time.Time, act *core.Tensor[T]) {
	if l == nil {
		return
	}
	cfg := tanhiCfg(l.Tanhi)
	if cfg == nil {
		return
	}
	cell := architecture.Cell{Layer: l.Core, Op: l, CombineMode: string(l.Cfg.Combine)}
	stampParallelTopology(&cell, l)
	tanhi.Emit(cfg, phase, idx, &cell, t0, t1, shapeFromTensorT(cfg, act))
}

func emitStackTanhi[T core.Numeric](s *Stack, phase string, idx int, t0, t1 time.Time, act *core.Tensor[T]) {
	if s == nil {
		return
	}
	cfg := tanhiCfg(s.Tanhi)
	if cfg == nil {
		return
	}
	cell := architecture.Cell{Layer: s.Core, Op: s}
	stampStackTopology(&cell, s)
	tanhi.Emit(cfg, phase, idx, &cell, t0, t1, shapeFromTensorT(cfg, act))
}
