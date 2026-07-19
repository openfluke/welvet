package convt1

import (
	"fmt"

	"github.com/openfluke/welvet/core"
)

// Config is ConvTranspose1d geometry (PyTorch / loom).
type Config struct {
	InChannels    int
	Filters       int
	SeqLen        int // input length
	Kernel        int
	Stride        int // 0 → 1
	Padding       int
	OutputPadding int // PyTorch output_padding; included in OutLen
	Activation    core.ActivationType
}

// OutputLength is ConvTranspose1d output length (PyTorch).
func OutputLength(seqLen, kernel, stride, padding, outputPadding int) int {
	if stride <= 0 {
		stride = 1
	}
	return (seqLen-1)*stride - 2*padding + kernel + outputPadding
}

// Validate fills defaults and checks geometry.
func (c *Config) Validate() error {
	if c == nil {
		return fmt.Errorf("convt1: nil config")
	}
	if c.InChannels <= 0 || c.Filters <= 0 || c.SeqLen <= 0 || c.Kernel <= 0 {
		return fmt.Errorf("convt1: need positive InChannels/Filters/SeqLen/Kernel")
	}
	if c.Stride <= 0 {
		c.Stride = 1
	}
	if c.Padding < 0 || c.OutputPadding < 0 {
		return fmt.Errorf("convt1: Padding/OutputPadding < 0")
	}
	if c.OutputPadding >= c.Stride {
		return fmt.Errorf("convt1: OutputPadding must be < Stride")
	}
	out := OutputLength(c.SeqLen, c.Kernel, c.Stride, c.Padding, c.OutputPadding)
	if out <= 0 {
		return fmt.Errorf("convt1: output length %d <= 0", out)
	}
	return nil
}

// OutLen returns validated output length.
func (c Config) OutLen() int {
	return OutputLength(c.SeqLen, c.Kernel, c.Stride, c.Padding, c.OutputPadding)
}

// PatchDim is InChannels × Kernel (Dense in-width).
func (c Config) PatchDim() int {
	return c.InChannels * c.Kernel
}
