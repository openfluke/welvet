package rmsnorm

import "fmt"

// Config is RMSNorm geometry.
type Config struct {
	Dim int
	Eps float64 // 0 → 1e-6
}

// Validate fills default eps.
func (c *Config) Validate() error {
	if c == nil {
		return fmt.Errorf("rmsnorm: nil config")
	}
	if c.Dim <= 0 {
		return fmt.Errorf("rmsnorm: need Dim>0")
	}
	if c.Eps <= 0 {
		c.Eps = 1e-6
	}
	return nil
}
