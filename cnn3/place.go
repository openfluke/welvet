package cnn3

import (
	"fmt"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
)

// Place binds this CNN3 layer onto the volumetric grid at (z,y,x,l).
func Place(g *architecture.Grid, z, y, x, lidx int, layer *Layer) error {
	if g == nil || layer == nil {
		return fmt.Errorf("cnn3: Place nil grid/layer")
	}
	layer.Core.Type = core.LayerCNN3
	layer.Core.Z, layer.Core.Y, layer.Core.X, layer.Core.L = z, y, x, lidx
	layer.Exec = g.Exec
	layer.syncProjExec()
	return g.BindOp(z, y, x, lidx, layer.Core, layer)
}
