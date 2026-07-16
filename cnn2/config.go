package cnn2

import (
	"fmt"

	"github.com/openfluke/welvet/core"
)

// Config is Conv2d geometry (PyTorch / loom; square kernel).
type Config struct {
	InChannels int
	Filters    int
	Height     int
	Width      int
	Kernel     int // kH = kW
	Stride     int // 0 → 1
	Padding    int
	Activation core.ActivationType
}

// OutputSpatial is the Conv2d output size along one axis (same as PyTorch).
func OutputSpatial(spatial, kernel, stride, padding int) int {
	if stride <= 0 {
		stride = 1
	}
	return (spatial+2*padding-kernel)/stride + 1
}

// Validate fills defaults and checks geometry.
func (c *Config) Validate() error {
	if c == nil {
		return fmt.Errorf("cnn2: nil config")
	}
	if c.InChannels <= 0 || c.Filters <= 0 || c.Height <= 0 || c.Width <= 0 || c.Kernel <= 0 {
		return fmt.Errorf("cnn2: need positive InChannels/Filters/Height/Width/Kernel")
	}
	if c.Stride <= 0 {
		c.Stride = 1
	}
	if c.Padding < 0 {
		return fmt.Errorf("cnn2: Padding < 0")
	}
	if c.OutH() <= 0 || c.OutW() <= 0 {
		return fmt.Errorf("cnn2: output spatial %dx%d invalid", c.OutH(), c.OutW())
	}
	return nil
}

// OutH / OutW are output spatial sizes.
func (c Config) OutH() int {
	return OutputSpatial(c.Height, c.Kernel, c.Stride, c.Padding)
}
func (c Config) OutW() int {
	return OutputSpatial(c.Width, c.Kernel, c.Stride, c.Padding)
}

// PatchDim is InChannels × Kernel × Kernel (Dense in-width).
func (c Config) PatchDim() int {
	return c.InChannels * c.Kernel * c.Kernel
}
