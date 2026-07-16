package swiglu

import "fmt"

// Config is SwiGLU geometry. Output dim equals InputDim (standard FFN).
type Config struct {
	InputDim        int
	IntermediateDim int
}

// Validate requires both dims.
func (c *Config) Validate() error {
	if c == nil {
		return fmt.Errorf("swiglu: nil config")
	}
	if c.InputDim <= 0 || c.IntermediateDim <= 0 {
		return fmt.Errorf("swiglu: need InputDim>0 and IntermediateDim>0")
	}
	return nil
}

// DefaultFFN returns a Llama-style ~8/3 intermediate (rounded to multiple of 8).
func DefaultFFN(dModel int) Config {
	inter := (dModel * 8) / 3
	inter = (inter + 7) / 8 * 8
	if inter < dModel {
		inter = dModel * 2
	}
	return Config{InputDim: dModel, IntermediateDim: inter}
}
