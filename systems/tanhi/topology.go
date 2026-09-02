package tanhi

import (
	"github.com/openfluke/welvet/architecture"
)

type topologyBridge func(cell *architecture.Cell)

type connectionCounter func(op any) int

var (
	topoBridge    topologyBridge
	gridTanhiSync func(g *architecture.Grid)
	connCounter   connectionCounter
)

// RegisterConnectionCounter counts flattened weights for tanhi connections field
// (registered from systems/dna to avoid tanhi→dna import cycle).
func RegisterConnectionCounter(fn connectionCounter) {
	connCounter = fn
}

// RegisterTopologyBridge fills Cell.ParallelBranches / SequentialLayers / CombineMode
// from cell.Op when topology metadata was not stamped at Place (avoids tanhi→parallel import cycle).
func RegisterTopologyBridge(fn topologyBridge) {
	topoBridge = fn
}

// RegisterGridTanhiSync propagates grid.Tanhi into nested cameral Ops after configure.
func RegisterGridTanhiSync(fn func(g *architecture.Grid)) {
	gridTanhiSync = fn
}

// SyncGridTanhi copies grid.Tanhi into Parallel / Stack / Sequential hosts on the grid.
func SyncGridTanhi(g *architecture.Grid) {
	if gridTanhiSync != nil && g != nil {
		gridTanhiSync(g)
	}
}

func enrichCellTopology(cell *architecture.Cell) {
	if topoBridge != nil && cell != nil {
		topoBridge(cell)
	}
}

// ConfigureGrid sets Tanhi on a grid and syncs into nested Ops.
func ConfigureGrid(g *architecture.Grid, cfg *UDPConfig) {
	if g == nil {
		return
	}
	g.Tanhi = cfg
	SyncGridTanhi(g)
}
