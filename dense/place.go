package dense

import (
	"fmt"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
)

// Place binds this Dense layer onto the volumetric grid at (z,y,x,l).
// The layer inherits the grid's ExecConfig (backend / tiling).
func Place(g *architecture.Grid, z, y, x, l int, layer *Layer) error {
	if g == nil || layer == nil {
		return fmt.Errorf("dense: Place nil grid/layer")
	}
	layer.Core.Type = core.LayerDense
	layer.Core.Z, layer.Core.Y, layer.Core.X, layer.Core.L = z, y, x, l
	layer.Exec = g.Exec
	return g.BindOp(z, y, x, l, layer.Core, layer)
}
