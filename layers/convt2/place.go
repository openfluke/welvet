package convt2

import (
	"fmt"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
)

func Place(g *architecture.Grid, z, y, x, lidx int, layer *Layer) error {
	if g == nil || layer == nil {
		return fmt.Errorf("convt2: Place nil grid/layer")
	}
	layer.Core.Type = core.LayerConvTransposed2D
	layer.Core.Z, layer.Core.Y, layer.Core.X, layer.Core.L = z, y, x, lidx
	layer.Exec = g.Exec
	layer.syncProjExec()
	return g.BindOp(z, y, x, lidx, layer.Core, layer)
}
