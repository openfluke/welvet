package residual

import (
	"fmt"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
)

// Place binds this Residual layer onto the volumetric grid at (z,y,x,l).
func Place(g *architecture.Grid, z, y, x, lidx int, layer *Layer) error {
	if g == nil || layer == nil {
		return fmt.Errorf("residual: Place nil grid/layer")
	}
	layer.Core.Type = core.LayerResidual
	layer.Core.Z, layer.Core.Y, layer.Core.X, layer.Core.L = z, y, x, lidx
	layer.Exec = g.Exec
	layer.syncChildExec()
	return g.BindOp(z, y, x, lidx, layer.Core, layer)
}
