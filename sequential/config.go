package sequential

import "fmt"

// Config is Sequential geometry — Depth Dense layers each Dim→Dim (shape-preserving).
type Config struct {
	Dim    int // feature width (in = out per child)
	SeqLen int // layout hint for [batch,seq,dim]
	Depth  int // number of Dense children; 0 → 2
}

// Validate fills defaults and checks geometry.
func (c *Config) Validate() error {
	if c == nil {
		return fmt.Errorf("sequential: nil config")
	}
	if c.Dim <= 0 {
		return fmt.Errorf("sequential: need positive Dim")
	}
	if c.SeqLen <= 0 {
		c.SeqLen = 1
	}
	if c.Depth <= 0 {
		c.Depth = 2
	}
	return nil
}

// WeightCount is Depth × Dim × Dim.
func (c Config) WeightCount() int {
	d := c.Depth
	if d <= 0 {
		d = 2
	}
	return d * c.Dim * c.Dim
}
