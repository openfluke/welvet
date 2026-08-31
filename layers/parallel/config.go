package parallel

import "fmt"

// CombineMode selects how branch outputs are merged.
type CombineMode string

const (
	CombineConcat   CombineMode = "concat"
	CombineAdd      CombineMode = "add"
	CombineAvg      CombineMode = "avg"
	CombineFilter   CombineMode = "filter"   // MoE: Dense gate → Softmax → weighted sum
	CombineMax      CombineMode = "max"      // elementwise max (hard route backward)
	CombineSparseK  CombineMode = "sparsek"  // top-K by ‖out‖₂, then avg
	CombineDisagree CombineMode = "disagree" // avg + β·(cam0−cam1) [2 cams] or avg+(self−mean)
)

// Config describes Parallel geometry. Dense New/NewConfigured use OutFeat per
// branch; NewFromBranches may leave OutFeat=0 and measure widths at forward.
type Config struct {
	Dim      int // input feature dim
	OutFeat  int // per-branch output feature dim (0 → measured dynamically)
	Branches int // number of branches (≥1)
	Combine  CombineMode
	SeqLen   int // 0 → treat input as [batch, Dim]; >0 → [batch, SeqLen, Dim]

	// SparseK is used by CombineSparseK (0 ⇒ min(2, Branches)).
	SparseK int
	// DisagreeBeta scales the disagreement term for CombineDisagree (0 ⇒ 1).
	DisagreeBeta float64
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
	case CombineConcat, CombineAdd, CombineAvg, CombineFilter,
		CombineMax, CombineSparseK, CombineDisagree:
	default:
		return fmt.Errorf("parallel: unknown Combine %q", c.Combine)
	}
	if c.SeqLen < 0 {
		return fmt.Errorf("parallel: SeqLen < 0")
	}
	if c.SparseK < 0 {
		return fmt.Errorf("parallel: SparseK < 0")
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

func (c Config) sparseK() int {
	k := c.SparseK
	if k <= 0 {
		k = 2
	}
	if k > c.Branches {
		k = c.Branches
	}
	if k < 1 {
		k = 1
	}
	return k
}

func (c Config) disagreeBeta() float64 {
	if c.DisagreeBeta == 0 {
		return 1
	}
	return c.DisagreeBeta
}
