package dna

import "github.com/openfluke/welvet/systems/tanhi"

func init() {
	tanhi.RegisterConnectionCounter(func(op any) int {
		flat, err := FlattenOp(op)
		if err != nil {
			return 0
		}
		return len(flat)
	})
}
