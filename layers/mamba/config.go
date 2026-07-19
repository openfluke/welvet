package mamba

import "fmt"

// Config is a minimal selective SSM.
type Config struct {
	DModel int
	DState int // N
	Expand int // 0 → 2; inner = Expand * DModel
	SeqLen int // expected T (validated on input)
}

// Validate fills defaults.
func (c *Config) Validate() error {
	if c == nil {
		return fmt.Errorf("mamba: nil config")
	}
	if c.DModel <= 0 || c.DState <= 0 {
		return fmt.Errorf("mamba: need positive DModel/DState")
	}
	if c.Expand <= 0 {
		c.Expand = 2
	}
	if c.SeqLen <= 0 {
		return fmt.Errorf("mamba: need positive SeqLen")
	}
	return nil
}

// InnerDim is Expand * DModel.
func (c Config) InnerDim() int { return c.Expand * c.DModel }
