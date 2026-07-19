package metacognition

import "fmt"

// Condition is what a heuristic checks on input/output stats.
type Condition int

const (
	CondNone Condition = iota
	CondStdAbove
	CondStdBelow
	CondAvgAbove
	CondAvgBelow
	CondMaxAbove
	CondActiveBelow
	CondGainDrift
)

// Effect applied when a rule fires (no topology morph).
type Effect int

const (
	EffectNone Effect = iota
	EffectGate       // scale output toward zero when unstable
	EffectIdentity   // reset Observed toward identity (square Dense)
	EffectPassthrough // skip Observed, return input
)

// Rule is if Condition(stats) crosses Threshold → Effect (with cooldown).
type Rule struct {
	Condition Condition
	Threshold float64
	Effect    Effect
	Cooldown  int
	fireCount int
}

// Config for Metacognition wrapper.
type Config struct {
	Dim    int
	SeqLen int // 0 → [batch, Dim]
	Rules  []Rule
}

// Validate checks Dim.
func (c *Config) Validate() error {
	if c == nil {
		return fmt.Errorf("metacognition: nil config")
	}
	if c.Dim <= 0 {
		return fmt.Errorf("metacognition: need positive Dim")
	}
	if c.SeqLen < 0 {
		return fmt.Errorf("metacognition: SeqLen < 0")
	}
	return nil
}

// DefaultStabilityRules returns practical gate/identity heuristics.
func DefaultStabilityRules() []Rule {
	return []Rule{
		{Condition: CondGainDrift, Threshold: 0.5, Effect: EffectIdentity, Cooldown: 5},
		{Condition: CondStdAbove, Threshold: 50.0, Effect: EffectGate, Cooldown: 3},
		{Condition: CondActiveBelow, Threshold: 0.01, Effect: EffectIdentity, Cooldown: 10},
	}
}
