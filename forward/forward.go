// Package forward walks the volumetric grid and dispatches cell ops (Dense, MHA, …).
//
// Contract: CPU tiled + SIMD + WebGPU via each layer's Exec; unsupported cell types
// hard-error (no silent skip of missing kernels). Tests live in w2a.
package forward

import (
	"fmt"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/dense"
	"github.com/openfluke/welvet/mha"
)

// Step records one executed cell for backward (and debugging).
type Step[T core.Numeric] struct {
	Coord architecture.Coord
	Input *core.Tensor[T]
	Pre   *core.Tensor[T]
	Post  *core.Tensor[T]
	Cell  *architecture.Cell
}

// Result is the network output plus the per-cell tape.
type Result[T core.Numeric] struct {
	Output *core.Tensor[T]
	Steps  []Step[T]
}

// Forward walks Depth→Rows→Cols→LayersPerCell (z→y→x→l), chaining activations.
// Remote-link cells take input from the target cell's recorded Post instead of
// the sequential previous activation.
func Forward[T core.Numeric](g *architecture.Grid, input *core.Tensor[T]) (*Result[T], error) {
	if g == nil || input == nil {
		return nil, fmt.Errorf("forward: nil grid/input")
	}
	current := input
	posts := make(map[architecture.Coord]*core.Tensor[T], g.StackLayerCount())
	var steps []Step[T]

	for _, coord := range g.HopOrder() {
		cell := g.At(coord.Z, coord.Y, coord.X, coord.L)
		if cell == nil || cell.Layer.IsDisabled {
			continue
		}
		if cell.Op == nil {
			return nil, fmt.Errorf("forward: no op at %v (type %s)", coord, cell.Layer.Type)
		}

		in := current
		if cell.IsRemoteLink {
			tc := architecture.Coord{Z: cell.TargetZ, Y: cell.TargetY, X: cell.TargetX, L: cell.TargetL}
			src, ok := posts[tc]
			if !ok || src == nil {
				return nil, fmt.Errorf("forward: remote hop %v → %v has no recorded activation", coord, tc)
			}
			in = src
		}

		pre, post, err := dispatch[T](cell, in)
		if err != nil {
			return nil, fmt.Errorf("forward %v: %w", coord, err)
		}
		steps = append(steps, Step[T]{Coord: coord, Input: in, Pre: pre, Post: post, Cell: cell})
		posts[coord] = post
		current = post
	}

	if len(steps) == 0 {
		return nil, fmt.Errorf("forward: grid has no enabled cells with ops")
	}
	return &Result[T]{Output: current, Steps: steps}, nil
}

func dispatch[T core.Numeric](cell *architecture.Cell, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	switch cell.Layer.Type {
	case core.LayerDense:
		dl, ok := cell.Op.(*dense.Layer)
		if !ok || dl == nil {
			return nil, nil, fmt.Errorf("dense cell Op is %T, want *dense.Layer", cell.Op)
		}
		return dense.Forward(dl, input)
	case core.LayerMultiHeadAttention:
		ml, ok := cell.Op.(*mha.Layer)
		if !ok || ml == nil {
			return nil, nil, fmt.Errorf("mha cell Op is %T, want *mha.Layer", cell.Op)
		}
		return mha.Forward(ml, input)
	default:
		return nil, nil, fmt.Errorf("unsupported layer type %s", cell.Layer.Type)
	}
}
