package convt3

import (
	"fmt"

	"github.com/openfluke/welvet/core"
)

// Config is ConvTranspose3d geometry (PyTorch / loom; cubic kernel).
type Config struct {
	InChannels    int
	Filters       int
	Depth         int
	Height        int
	Width         int
	Kernel        int
	Stride        int // 0 → 1
	Padding       int
	OutputPadding int
	Activation    core.ActivationType
}

func OutputSpatial(spatial, kernel, stride, padding, outputPadding int) int {
	if stride <= 0 {
		stride = 1
	}
	return (spatial-1)*stride - 2*padding + kernel + outputPadding
}

func (c *Config) Validate() error {
	if c == nil {
		return fmt.Errorf("convt3: nil config")
	}
	if c.InChannels <= 0 || c.Filters <= 0 || c.Depth <= 0 || c.Height <= 0 || c.Width <= 0 || c.Kernel <= 0 {
		return fmt.Errorf("convt3: need positive InChannels/Filters/Depth/Height/Width/Kernel")
	}
	if c.Stride <= 0 {
		c.Stride = 1
	}
	if c.Padding < 0 || c.OutputPadding < 0 {
		return fmt.Errorf("convt3: Padding/OutputPadding < 0")
	}
	if c.OutputPadding >= c.Stride {
		return fmt.Errorf("convt3: OutputPadding must be < Stride")
	}
	if c.OutD() <= 0 || c.OutH() <= 0 || c.OutW() <= 0 {
		return fmt.Errorf("convt3: output spatial %dx%dx%d invalid", c.OutD(), c.OutH(), c.OutW())
	}
	return nil
}

func (c Config) OutD() int {
	return OutputSpatial(c.Depth, c.Kernel, c.Stride, c.Padding, c.OutputPadding)
}
func (c Config) OutH() int {
	return OutputSpatial(c.Height, c.Kernel, c.Stride, c.Padding, c.OutputPadding)
}
func (c Config) OutW() int {
	return OutputSpatial(c.Width, c.Kernel, c.Stride, c.Padding, c.OutputPadding)
}

func (c Config) PatchDim() int {
	return c.InChannels * c.Kernel * c.Kernel * c.Kernel
}
