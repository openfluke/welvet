// Package placement applies the shared JSON/native execution configuration to
// existing graph operations. It changes placement, never graph connections.
package placement

import (
	"fmt"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/nativecpp"
	"strconv"
)

type configurable interface {
	ApplyExecConfig(nativecpp.ExecConfig, string, string, string)
}

// Stack assigns each immediate child using its zero-based layer index. A child
// Stack/Parallel is a segment; its explicitly pinned descendants keep overrides.
func Stack(s *parallel.Stack, cfg nativecpp.ExecConfig, volume, sector string) error {
	if s == nil {
		return fmt.Errorf("placement: nil Stack")
	}
	for _, op := range s.Children {
		if _, ok := op.(configurable); !ok {
			return fmt.Errorf("placement: unsupported operation %T", op)
		}
	}
	for i, op := range s.Children {
		op.(configurable).ApplyExecConfig(cfg, volume, sector, strconv.Itoa(i))
	}
	return nil
}

// Grid uses sector keys "z,y,x,0" and layer keys L, matching the existing
// four-coordinate sector convention. Remote links and disabled cells are kept.
func Grid(g *architecture.Grid, cfg nativecpp.ExecConfig, volume string) error {
	if g == nil {
		return fmt.Errorf("placement: nil Grid")
	}
	for _, c := range g.Cells {
		if c.Layer.IsDisabled {
			continue
		}
		if _, ok := c.Op.(configurable); !ok {
			return fmt.Errorf("placement: unsupported operation %T", c.Op)
		}
	}
	for _, coord := range g.HopOrder() {
		c := g.At(coord.Z, coord.Y, coord.X, coord.L)
		if c.Layer.IsDisabled {
			continue
		}
		c.Op.(configurable).ApplyExecConfig(cfg, volume, fmt.Sprintf("%d,%d,%d,0", coord.Z, coord.Y, coord.X), strconv.Itoa(coord.L))
	}
	return nil
}
