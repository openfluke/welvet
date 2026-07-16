// Package architecture owns the volumetric grid: Depth × Rows × Cols × LayersPerCell.
//
// Mirrors loom/poly VolumetricNetwork topology (cells, hops, remote links) without
// baking in layer compute — each Cell holds a core.Layer descriptor; Dense/MHA/…
// packages own execution. Contract: CPU tiled + SIMD + WebGPU × dtype × k-quant
// at the cell payload level. Tests live in github.com/openfluke/w2a.
package architecture

import "github.com/openfluke/welvet/core"

// Coord is a volumetric address (Z depth, Y row, X col, L layer-in-cell).
type Coord struct {
	Z, Y, X, L int
}

// Cell is one processing unit in the grid (loom VolumetricLayer topology subset).
type Cell struct {
	Layer core.Layer

	// Spatial hop: when IsRemoteLink, inputs come from Target* instead of local prior.
	IsRemoteLink bool
	TargetZ      int
	TargetY      int
	TargetX      int
	TargetL      int

	ParallelBranches []Cell
	SequentialLayers []Cell
	CombineMode      string // "concat", "add", "avg", …
}

// Grid is the volumetric network container (Depth × Rows × Cols × LayersPerCell).
type Grid struct {
	Depth         int
	Rows          int
	Cols          int
	LayersPerCell int
	Cells         []Cell
	Exec          core.ExecConfig
	NativeExact   bool
}

// VolumetricNetwork is an alias matching loom/poly naming.
type VolumetricNetwork = Grid

// NewGrid allocates a volumetric grid and stamps default Dense+ReLU+Float32 cells.
func NewGrid(depth, rows, cols, layersPerCell int) *Grid {
	if depth < 1 {
		depth = 1
	}
	if rows < 1 {
		rows = 1
	}
	if cols < 1 {
		cols = 1
	}
	if layersPerCell < 1 {
		layersPerCell = 1
	}
	total := depth * rows * cols * layersPerCell
	g := &Grid{
		Depth:         depth,
		Rows:          rows,
		Cols:          cols,
		LayersPerCell: layersPerCell,
		Cells:         make([]Cell, total),
		NativeExact:   true,
		Exec: core.ExecConfig{
			Backend:   core.BackendCPUTiled,
			MultiCore: true,
			TileSize:  32,
		},
	}
	for z := 0; z < depth; z++ {
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				for l := 0; l < layersPerCell; l++ {
					idx := g.Index(z, y, x, l)
					g.Cells[idx] = Cell{
						Layer: core.Layer{
							Type:         core.LayerDense,
							DType:        core.DTypeFloat32,
							Activation:   core.ActivationReLU,
							Z:            z,
							Y:            y,
							X:            x,
							L:            l,
							TileSize:     32,
							MultiCore:    true,
						},
					}
				}
			}
		}
	}
	return g
}

// NewVolumetricNetwork matches loom/poly constructor name.
func NewVolumetricNetwork(depth, rows, cols, layersPerCell int) *Grid {
	return NewGrid(depth, rows, cols, layersPerCell)
}

// Index flattens (z,y,x,l) → linear cell index.
func (g *Grid) Index(z, y, x, l int) int {
	if g == nil {
		return -1
	}
	return (((z*g.Rows+y)*g.Cols+x)*g.LayersPerCell + l)
}

// At returns the cell at coordinates, or nil if out of range.
func (g *Grid) At(z, y, x, l int) *Cell {
	if g == nil {
		return nil
	}
	idx := g.Index(z, y, x, l)
	if idx < 0 || idx >= len(g.Cells) {
		return nil
	}
	return &g.Cells[idx]
}

// GetLayer is the loom-compatible alias for At.
func (g *Grid) GetLayer(z, y, x, l int) *Cell { return g.At(z, y, x, l) }

// StackLayerCount returns Depth*Rows*Cols*LayersPerCell.
func (g *Grid) StackLayerCount() int {
	if g == nil {
		return 0
	}
	return g.Depth * g.Rows * g.Cols * g.LayersPerCell
}

// HopOrder returns traversal order z→y→x→l (each entry is one hop).
func (g *Grid) HopOrder() []Coord {
	if g == nil {
		return nil
	}
	out := make([]Coord, 0, g.StackLayerCount())
	for z := 0; z < g.Depth; z++ {
		for y := 0; y < g.Rows; y++ {
			for x := 0; x < g.Cols; x++ {
				for l := 0; l < g.LayersPerCell; l++ {
					out = append(out, Coord{Z: z, Y: y, X: x, L: l})
				}
			}
		}
	}
	return out
}

// SetRemoteLink marks cell (z,y,x,l) as a spatial hop to target.
func (g *Grid) SetRemoteLink(z, y, x, l, tz, ty, tx, tl int) error {
	c := g.At(z, y, x, l)
	if c == nil {
		return errCoord("source", z, y, x, l)
	}
	if g.At(tz, ty, tx, tl) == nil {
		return errCoord("target", tz, ty, tx, tl)
	}
	c.IsRemoteLink = true
	c.TargetZ, c.TargetY, c.TargetX, c.TargetL = tz, ty, tx, tl
	return nil
}

// ResolveHop returns the effective source cell for a hop (remote target or self).
func (g *Grid) ResolveHop(c *Cell) *Cell {
	if g == nil || c == nil {
		return nil
	}
	if c.IsRemoteLink {
		return g.At(c.TargetZ, c.TargetY, c.TargetX, c.TargetL)
	}
	return c
}

// ClearRemoteLink clears a spatial hop on the cell.
func (g *Grid) ClearRemoteLink(z, y, x, l int) {
	if c := g.At(z, y, x, l); c != nil {
		c.IsRemoteLink = false
		c.TargetZ, c.TargetY, c.TargetX, c.TargetL = 0, 0, 0, 0
	}
}
