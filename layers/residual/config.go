package residual

import "fmt"

// Config is Residual geometry — Depth Dense layers as F, each Dim→Dim, then +x.
type Config struct {
	Dim    int // feature width (in = out)
	SeqLen int // layout hint for [batch,seq,dim]
	Depth  int // Dense children in F; 0 → 1
}

// Validate fills defaults and checks geometry.
func (c *Config) Validate() error {
	if c == nil {
		return fmt.Errorf("residual: nil config")
	}
	if c.Dim <= 0 {
		return fmt.Errorf("residual: need positive Dim")
	}
	if c.SeqLen <= 0 {
		c.SeqLen = 1
	}
	if c.Depth <= 0 {
		c.Depth = 1
	}
	return nil
}

// WeightCount is Depth × Dim × Dim.
func (c Config) WeightCount() int {
	d := c.Depth
	if d <= 0 {
		d = 1
	}
	return d * c.Dim * c.Dim
}
