package cnn1

import (
	"fmt"

	"github.com/openfluke/welvet/core"
)

// Config is Conv1d geometry (PyTorch / loom).
type Config struct {
	InChannels int
	Filters    int
	SeqLen     int // expected input length
	Kernel     int
	Stride     int // 0 → 1
	Padding    int
	Activation core.ActivationType
}

// OutputLength is the Conv1d output length (same as PyTorch).
func OutputLength(seqLen, kernel, stride, padding int) int {
	if stride <= 0 {
		stride = 1
	}
	return (seqLen+2*padding-kernel)/stride + 1
}

// Validate fills defaults and checks geometry.
func (c *Config) Validate() error {
	if c == nil {
		return fmt.Errorf("cnn1: nil config")
	}
	if c.InChannels <= 0 || c.Filters <= 0 || c.SeqLen <= 0 || c.Kernel <= 0 {
		return fmt.Errorf("cnn1: need positive InChannels/Filters/SeqLen/Kernel")
	}
	if c.Stride <= 0 {
		c.Stride = 1
	}
	if c.Padding < 0 {
		return fmt.Errorf("cnn1: Padding < 0")
	}
	out := OutputLength(c.SeqLen, c.Kernel, c.Stride, c.Padding)
	if out <= 0 {
		return fmt.Errorf("cnn1: output length %d <= 0", out)
	}
	return nil
}

// OutLen returns validated output length.
func (c Config) OutLen() int {
	return OutputLength(c.SeqLen, c.Kernel, c.Stride, c.Padding)
}

// PatchDim is InChannels × Kernel (Dense in-width).
func (c Config) PatchDim() int {
	return c.InChannels * c.Kernel
}
