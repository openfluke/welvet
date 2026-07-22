package parallel

import "fmt"

// CombineMode selects how branch outputs are merged.
type CombineMode string

const (
	CombineConcat CombineMode = "concat"
	CombineAdd    CombineMode = "add"
	CombineAvg    CombineMode = "avg"
	CombineFilter CombineMode = "filter" // MoE: Dense gate → Softmax → weighted sum
)

// Config describes Parallel geometry. Dense New/NewConfigured use OutFeat per
// branch; NewFromBranches may leave OutFeat=0 and measure widths at forward.
type Config struct {
	Dim      int // input feature dim
	OutFeat  int // per-branch output feature dim (0 → measured dynamically)
	Branches int // number of branches (≥1)
	Combine  CombineMode
	SeqLen   int // 0 → treat input as [batch, Dim]; >0 → [batch, SeqLen, Dim]
}

// Validate fills defaults. OutFeat may be 0 for polymorphic NewFromBranches.
func (c *Config) Validate() error {
	if c == nil {
		return fmt.Errorf("parallel: nil config")
	}
	if c.Dim <= 0 || c.Branches <= 0 {
		return fmt.Errorf("parallel: need positive Dim/Branches")
	}
	if c.OutFeat < 0 {
		return fmt.Errorf("parallel: OutFeat < 0")
	}
	if c.Combine == "" {
		c.Combine = CombineConcat
	}
	switch c.Combine {
	case CombineConcat, CombineAdd, CombineAvg, CombineFilter:
	default:
		return fmt.Errorf("parallel: unknown Combine %q", c.Combine)
	}
	if c.SeqLen < 0 {
		return fmt.Errorf("parallel: SeqLen < 0")
	}
	return nil
}

// OutDim is the combined feature dimension (0 when OutFeat is dynamic/unset).
func (c Config) OutDim() int {
	if c.OutFeat <= 0 {
		return 0
	}
	switch c.Combine {
	case CombineConcat:
		return c.Branches * c.OutFeat
	default:
		return c.OutFeat
	}
}
