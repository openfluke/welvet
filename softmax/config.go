package softmax

import "fmt"

// Kind selects Softmax geometry.
type Kind int

const (
	// KindStandard — independent Softmax over the last axis of the input.
	KindStandard Kind = iota
	// KindGrid — Softmax over Rows groups of Cols (flat layout).
	KindGrid
)

func (k Kind) String() string {
	switch k {
	case KindStandard:
		return "standard"
	case KindGrid:
		return "grid"
	default:
		return fmt.Sprintf("Kind(%d)", int(k))
	}
}

// Config is Softmax geometry / temperature.
type Config struct {
	Dim         int     // last-axis size (classes); also default Cols
	SeqLen      int     // layout hint (tokens / width for smokes)
	Rows        int     // KindGrid only; 0 → derive
	Cols        int     // KindGrid only; 0 → Dim
	Temperature float64 // 0 → 1
	Kind        Kind
}

// Validate fills defaults and checks geometry.
func (c *Config) Validate() error {
	if c == nil {
		return fmt.Errorf("softmax: nil config")
	}
	if c.Dim <= 0 {
		return fmt.Errorf("softmax: need positive Dim")
	}
	if c.SeqLen <= 0 {
		c.SeqLen = 1
	}
	if c.Temperature == 0 {
		c.Temperature = 1
	}
	if c.Temperature < 0 {
		return fmt.Errorf("softmax: Temperature < 0")
	}
	if c.Kind == KindGrid {
		if c.Cols <= 0 {
			c.Cols = c.Dim
		}
		if c.Rows <= 0 {
			return fmt.Errorf("softmax: KindGrid needs positive Rows")
		}
	}
	return nil
}

// WeightCount is always 0 (weightless).
func (c Config) WeightCount() int { return 0 }
