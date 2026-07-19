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

// Config describes Parallel geometry. Branches are Dim→Dim Dense (OutFeat each);
// for concat, combined out = Branches * OutFeat; for add/avg/filter OutFeat must match.
type Config struct {
	Dim      int // input feature dim
	OutFeat  int // per-branch output feature dim
	Branches int // number of Dense branches (≥1)
	Combine  CombineMode
	SeqLen   int // 0 → treat input as [batch, Dim]; >0 → [batch, SeqLen, Dim]
}

// Validate fills defaults.
func (c *Config) Validate() error {
	if c == nil {
		return fmt.Errorf("parallel: nil config")
	}
	if c.Dim <= 0 || c.OutFeat <= 0 || c.Branches <= 0 {
		return fmt.Errorf("parallel: need positive Dim/OutFeat/Branches")
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

// OutDim is the combined feature dimension.
func (c Config) OutDim() int {
	switch c.Combine {
	case CombineConcat:
		return c.Branches * c.OutFeat
	default:
		return c.OutFeat
	}
}
