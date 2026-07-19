package convt2

import (
	"fmt"

	"github.com/openfluke/welvet/core"
)

// Config is ConvTranspose2d geometry (PyTorch / loom; square kernel).
type Config struct {
	InChannels    int
	Filters       int
	Height        int
	Width         int
	Kernel        int
	Stride        int // 0 → 1
	Padding       int
	OutputPadding int
	Activation    core.ActivationType
}

// OutputSpatial is ConvTranspose2d size along one axis (PyTorch).
func OutputSpatial(spatial, kernel, stride, padding, outputPadding int) int {
	if stride <= 0 {
		stride = 1
	}
	return (spatial-1)*stride - 2*padding + kernel + outputPadding
}

// Validate fills defaults and checks geometry.
func (c *Config) Validate() error {
	if c == nil {
		return fmt.Errorf("convt2: nil config")
	}
	if c.InChannels <= 0 || c.Filters <= 0 || c.Height <= 0 || c.Width <= 0 || c.Kernel <= 0 {
		return fmt.Errorf("convt2: need positive InChannels/Filters/Height/Width/Kernel")
	}
	if c.Stride <= 0 {
		c.Stride = 1
	}
	if c.Padding < 0 || c.OutputPadding < 0 {
		return fmt.Errorf("convt2: Padding/OutputPadding < 0")
	}
	if c.OutputPadding >= c.Stride {
		return fmt.Errorf("convt2: OutputPadding must be < Stride")
	}
	if c.OutH() <= 0 || c.OutW() <= 0 {
		return fmt.Errorf("convt2: output spatial %dx%d invalid", c.OutH(), c.OutW())
	}
	return nil
}

func (c Config) OutH() int {
	return OutputSpatial(c.Height, c.Kernel, c.Stride, c.Padding, c.OutputPadding)
}
func (c Config) OutW() int {
	return OutputSpatial(c.Width, c.Kernel, c.Stride, c.Padding, c.OutputPadding)
}

// PatchDim is InChannels × Kernel × Kernel.
func (c Config) PatchDim() int {
	return c.InChannels * c.Kernel * c.Kernel
}
