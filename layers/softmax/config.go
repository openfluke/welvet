package softmax

import "fmt"

// Kind selects Softmax geometry / variant.
type Kind int

const (
	// KindStandard — independent Softmax over the last axis of the input.
	KindStandard Kind = iota
	// KindTemperature — standard with explicit temperature scaling.
	KindTemperature
	// KindGumbel — Gumbel-Softmax (uses math/rand; non-deterministic).
	KindGumbel
	// KindMasked — masked positions set to −1e9 before Softmax.
	KindMasked
	// KindSparse — sparsemax over each group.
	KindSparse
	// KindEntmax — entmax-α (EntmaxAlpha; 0 → 1.5).
	KindEntmax
	// KindGrid — Softmax over Rows groups of Cols (flat layout).
	KindGrid
	// KindHierarchical — grid with HierarchyLevels last entry as cols.
	KindHierarchical
)

func (k Kind) String() string {
	switch k {
	case KindStandard:
		return "standard"
	case KindTemperature:
		return "temperature"
	case KindGumbel:
		return "gumbel"
	case KindMasked:
		return "masked"
	case KindSparse:
		return "sparse"
	case KindEntmax:
		return "entmax"
	case KindGrid:
		return "grid"
	case KindHierarchical:
		return "hierarchical"
	default:
		return fmt.Sprintf("Kind(%d)", int(k))
	}
}

// ParseKind maps a serialized kind string.
func ParseKind(s string) Kind {
	switch s {
	case "grid":
		return KindGrid
	case "temperature":
		return KindTemperature
	case "gumbel":
		return KindGumbel
	case "masked":
		return KindMasked
	case "sparse":
		return KindSparse
	case "entmax":
		return KindEntmax
	case "hierarchical":
		return KindHierarchical
	default:
		return KindStandard
	}
}

// Config is Softmax geometry / temperature / variant options.
type Config struct {
	Dim             int     // last-axis size (classes); also default Cols
	SeqLen          int     // layout hint (tokens / width for smokes)
	Rows            int     // KindGrid / KindHierarchical; 0 → derive
	Cols            int     // KindGrid; 0 → Dim
	Temperature     float64 // 0 → 1
	Kind            Kind
	Mask            []bool    // KindMasked: false → masked out
	EntmaxAlpha     float64   // KindEntmax; 0 → 1.5
	HierarchyLevels []int     // KindHierarchical: last level is cols
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
	switch c.Kind {
	case KindGrid:
		if c.Cols <= 0 {
			c.Cols = c.Dim
		}
		if c.Rows <= 0 {
			return fmt.Errorf("softmax: KindGrid needs positive Rows")
		}
	case KindHierarchical:
		if len(c.HierarchyLevels) == 0 {
			return fmt.Errorf("softmax: KindHierarchical needs HierarchyLevels")
		}
		if c.Cols <= 0 {
			c.Cols = c.HierarchyLevels[len(c.HierarchyLevels)-1]
		}
		if c.Rows <= 0 {
			// Rows derived at runtime from tensor length.
		}
	}
	return nil
}

// WeightCount is always 0 (weightless).
func (c Config) WeightCount() int { return 0 }
