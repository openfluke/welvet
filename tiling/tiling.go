package tiling

// Package tiling chooses Dense/layer tile sizes for CPU SC/MC and GPU workgroups.
// Caps are hardware-aware defaults; callers may override via ExecConfig.TileSize.

const (
	DefaultCPUTile   = 32
	DefaultGPUWG     = 64
	MinCPUTile       = 8
	MaxCPUTile       = 256
)

// CPUTile returns a clamped tile size for CPU tiled Dense (and similar GEMVs).
func CPUTile(requested int) int {
	if requested <= 0 {
		return DefaultCPUTile
	}
	if requested < MinCPUTile {
		return MinCPUTile
	}
	if requested > MaxCPUTile {
		return MaxCPUTile
	}
	return requested
}

// PreferMultiCore reports whether batch*out is large enough to pay for goroutine fan-out.
func PreferMultiCore(batch, out, tile int) bool {
	tile = CPUTile(tile)
	return batch*out > tile
}

// GPUWorkgroupsX returns dispatch X for one-thread-per-output GEMV.
func GPUWorkgroupsX(out, workgroupSize int) uint32 {
	if workgroupSize <= 0 {
		workgroupSize = DefaultGPUWG
	}
	if out <= 0 {
		return 0
	}
	return uint32((out + workgroupSize - 1) / workgroupSize)
}
