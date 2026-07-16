package cnn3

import (
	"fmt"

	"github.com/openfluke/welvet/core"
)

// Config is Conv3d geometry (PyTorch / loom; cubic kernel).
type Config struct {
	InChannels int
	Filters    int
	Depth      int
	Height     int
	Width      int
	Kernel     int // kD = kH = kW
	Stride     int // 0 → 1
	Padding    int
	Activation core.ActivationType
}

// OutputSpatial is the Conv3d output size along one axis (same as PyTorch).
func OutputSpatial(spatial, kernel, stride, padding int) int {
	if stride <= 0 {
		stride = 1
	}
	return (spatial+2*padding-kernel)/stride + 1
}

// Validate fills defaults and checks geometry.
func (c *Config) Validate() error {
	if c == nil {
		return fmt.Errorf("cnn3: nil config")
	}
	if c.InChannels <= 0 || c.Filters <= 0 || c.Depth <= 0 || c.Height <= 0 || c.Width <= 0 || c.Kernel <= 0 {
		return fmt.Errorf("cnn3: need positive InChannels/Filters/Depth/Height/Width/Kernel")
	}
	if c.Stride <= 0 {
		c.Stride = 1
	}
	if c.Padding < 0 {
		return fmt.Errorf("cnn3: Padding < 0")
	}
	if c.OutD() <= 0 || c.OutH() <= 0 || c.OutW() <= 0 {
		return fmt.Errorf("cnn3: output spatial %dx%dx%d invalid", c.OutD(), c.OutH(), c.OutW())
	}
	return nil
}

// OutD / OutH / OutW are output spatial sizes.
func (c Config) OutD() int {
	return OutputSpatial(c.Depth, c.Kernel, c.Stride, c.Padding)
}
func (c Config) OutH() int {
	return OutputSpatial(c.Height, c.Kernel, c.Stride, c.Padding)
}
func (c Config) OutW() int {
	return OutputSpatial(c.Width, c.Kernel, c.Stride, c.Padding)
}

// PatchDim is InChannels × Kernel³ (Dense in-width).
func (c Config) PatchDim() int {
	return c.InChannels * c.Kernel * c.Kernel * c.Kernel
}
