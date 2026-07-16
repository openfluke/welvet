package architecture

import (
	"fmt"

	"github.com/openfluke/welvet/core"
)

// SetOp attaches an executable payload and syncs the cell's core.Layer descriptor.
func (c *Cell) SetOp(meta core.Layer, op any) {
	if c == nil {
		return
	}
	meta.Z, meta.Y, meta.X, meta.L = c.Layer.Z, c.Layer.Y, c.Layer.X, c.Layer.L
	c.Layer = meta
	c.Op = op
}

// BindOp places op at (z,y,x,l). meta should describe Type/DType/geometry/activation.
func (g *Grid) BindOp(z, y, x, l int, meta core.Layer, op any) error {
	c := g.At(z, y, x, l)
	if c == nil {
		return errCoord("bind", z, y, x, l)
	}
	if op == nil {
		return fmt.Errorf("architecture: nil op at z=%d y=%d x=%d l=%d", z, y, x, l)
	}
	c.SetOp(meta, op)
	return nil
}
