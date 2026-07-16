package layernorm

import "fmt"

// Config is LayerNorm geometry.
type Config struct {
	Dim int
	Eps float64 // 0 → 1e-5 (loom default)
}

// Validate fills default eps.
func (c *Config) Validate() error {
	if c == nil {
		return fmt.Errorf("layernorm: nil config")
	}
	if c.Dim <= 0 {
		return fmt.Errorf("layernorm: need Dim>0")
	}
	if c.Eps <= 0 {
		c.Eps = 1e-5
	}
	return nil
}
