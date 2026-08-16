package parallel

import (
	"fmt"

	"github.com/openfluke/welvet/core"
)

// ResidualSkip wraps F so y = F(x) + x. Sequential/Residual cannot import
// this package, so Parallel-as-F lives here (scorecard nested residual graft).
type ResidualSkip struct {
	F    any
	Exec core.ExecConfig
}

// ResidualGraft builds a shape-preserving skip around F (Dense, Sequential,
// Residual, Parallel CombineAdd/Avg, …). F(x) must match x's element count.
func ResidualGraft(f any) (*ResidualSkip, error) {
	if f == nil {
		return nil, fmt.Errorf("parallel: ResidualGraft nil F")
	}
	return &ResidualSkip{F: f}, nil
}

func skipAdd[T core.Numeric](fx, x *core.Tensor[T]) (*core.Tensor[T], error) {
	if fx == nil || x == nil {
		return nil, fmt.Errorf("parallel: ResidualSkip nil tensor")
	}
	if fx.Len() != x.Len() {
		return nil, fmt.Errorf("parallel: ResidualSkip F(x) len %d != x len %d (F must preserve shape)", fx.Len(), x.Len())
	}
	y := core.NewTensor[T](fx.Shape...)
	for i := range y.Data {
		y.Data[i] = core.FromFloat64[T](core.AsFloat64(fx.Data[i]) + core.AsFloat64(x.Data[i]))
	}
	return y, nil
}

func residualSkipForward[T core.Numeric](v *ResidualSkip, input, flat *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if v == nil || v.F == nil {
		return nil, nil, fmt.Errorf("parallel: nil ResidualSkip")
	}
	pre, fx, err := branchForward(v.F, input, flat)
	if err != nil {
		return nil, nil, err
	}
	skip := input
	if fx != nil && input != nil && fx.Len() != input.Len() && flat != nil && fx.Len() == flat.Len() {
		skip = flat
	}
	post, err = skipAdd(fx, skip)
	return pre, post, err
}

func residualSkipBackward[T core.Numeric](v *ResidualSkip, gradOut, input, flat, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	if v == nil || v.F == nil {
		return nil, nil, fmt.Errorf("parallel: nil ResidualSkip")
	}
	gF, dW, err := branchBackward(v.F, gradOut, input, flat, pre)
	if err != nil {
		return nil, nil, err
	}
	skip := gradOut
	if gF != nil && gradOut != nil && gF.Len() != gradOut.Len() && flat != nil && gF.Len() == flat.Len() {
		skip = flat
	}
	gradIn, err = skipAdd(gF, skip)
	return gradIn, dW, err
}
