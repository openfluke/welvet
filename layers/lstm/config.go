package lstm

import "fmt"

// Config is LSTM geometry.
type Config struct {
	InputSize  int
	HiddenSize int
	SeqLen     int
}

// Validate checks geometry.
func (c *Config) Validate() error {
	if c == nil {
		return fmt.Errorf("lstm: nil config")
	}
	if c.InputSize <= 0 || c.HiddenSize <= 0 || c.SeqLen <= 0 {
		return fmt.Errorf("lstm: need positive InputSize/HiddenSize/SeqLen")
	}
	return nil
}

// GateSize is len(ih)+len(hh)+len(bias) for one gate.
func (c Config) GateSize() int {
	h, in := c.HiddenSize, c.InputSize
	return h*in + h*h + h
}

// WeightCount is 4 × GateSize (i,f,g,o loom pack).
func (c Config) WeightCount() int {
	return 4 * c.GateSize()
}
