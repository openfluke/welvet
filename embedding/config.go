package embedding

import "fmt"

// Config is embedding geometry.
type Config struct {
	VocabSize    int
	EmbeddingDim int
	SeqLen       int // expected sequence length (layout hint)
}

// Validate checks geometry.
func (c *Config) Validate() error {
	if c == nil {
		return fmt.Errorf("embedding: nil config")
	}
	if c.VocabSize <= 0 || c.EmbeddingDim <= 0 || c.SeqLen <= 0 {
		return fmt.Errorf("embedding: need positive VocabSize/EmbeddingDim/SeqLen")
	}
	return nil
}

// WeightCount is vocab × emb.
func (c Config) WeightCount() int {
	return c.VocabSize * c.EmbeddingDim
}
