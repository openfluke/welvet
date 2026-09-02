package sequential

import (
	"time"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/systems/tanhi"
)

func emitSequentialTanhi[T core.Numeric](l *Layer, phase string, idx int, t0, t1 time.Time, act *core.Tensor[T]) {
	if l == nil {
		return
	}
	cfg := tanhi.ConfigFromAny(l.Tanhi)
	if cfg == nil {
		return
	}
	cell := architecture.Cell{Layer: l.Core, Op: l}
	children := l.ChildOps()
	if len(children) > 0 {
		z, y, x, parentL := l.Core.Z, l.Core.Y, l.Core.X, l.Core.L
		layers := make([]architecture.Cell, len(children))
		for i, op := range children {
			meta := childLayerMeta(op, l.Core)
			meta.Z, meta.Y, meta.X, meta.L = z, y, x, parentL
			layers[i] = architecture.Cell{Layer: meta, Op: op}
		}
		cell.SequentialLayers = layers
	}
	var shape []int
	if cfg.SendShape && act != nil {
		shape = append([]int(nil), act.Shape...)
	}
	tanhi.Emit(cfg, phase, idx, &cell, t0, t1, shape)
}

func childLayerMeta(op any, fallback core.Layer) core.Layer {
	if d, ok := op.(*dense.Layer); ok && d != nil {
		return d.Core
	}
	return fallback
}
