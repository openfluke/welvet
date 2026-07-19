package rnn

import "fmt"

// Config is vanilla RNN geometry.
type Config struct {
	InputSize  int
	HiddenSize int
	SeqLen     int // expected sequence length
}

// Validate checks geometry.
func (c *Config) Validate() error {
	if c == nil {
		return fmt.Errorf("rnn: nil config")
	}
	if c.InputSize <= 0 || c.HiddenSize <= 0 || c.SeqLen <= 0 {
		return fmt.Errorf("rnn: need positive InputSize/HiddenSize/SeqLen")
	}
	return nil
}

// WeightCount is len(W_ih)+len(W_hh)+len(bias) (loom pack).
func (c Config) WeightCount() int {
	h, in := c.HiddenSize, c.InputSize
	return h*in + h*h + h
}
