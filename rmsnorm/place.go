package rmsnorm

import (
	"fmt"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
)

// Place binds this RMSNorm layer onto the volumetric grid at (z,y,x,l).
func Place(g *architecture.Grid, z, y, x, lidx int, layer *Layer) error {
	if g == nil || layer == nil {
		return fmt.Errorf("rmsnorm: Place nil grid/layer")
	}
	layer.Core.Type = core.LayerRMSNorm
	layer.Core.Z, layer.Core.Y, layer.Core.X, layer.Core.L = z, y, x, lidx
	layer.Exec = g.Exec
	return g.BindOp(z, y, x, lidx, layer.Core, layer)
}
