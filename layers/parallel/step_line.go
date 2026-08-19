package parallel

import (
	"fmt"

	"github.com/openfluke/welvet/core"
)

// Line is a systolic 1D pipe: one new sample at layer 0 per Tick, every
// in-flight sample advances one child. Output at depth D is the sample that
// entered D ticks ago. Train when that sample pops the last child, using the
// activations saved along its path (not a full-chain re-forward of the latest x).
type Line[T core.Numeric] struct {
	flights []lineFlight[T]
}

type lineFlight[T core.Numeric] struct {
	target           *core.Tensor[T]
	stage            int
	ins, pres, posts []*core.Tensor[T]
}

// IsLineStep is the real step schedule: queue + one layer hop per TrainStep.
// Mesh* stays grid-only (RequiresGrid). Normal/Tween without Step prefix is full-chain.
func (m TrainMode) IsLineStep() bool {
	switch m.Resolve(ModeNormalBP) {
	case ModeStepBP, ModeStepTween, ModeStepTweenChain, ModeStepTweenSplit, ModeStepTweenAlt,
		ModeStepTweenSplitHeadProxy, ModeStepTweenSplitLinear, ModeStepTweenSplitFastProxy,
		ModeStepTweenSplitLinearCache, ModeStepTweenSplitHeadProxyAsync, ModeStepTweenSplitSparse:
		return true
	default:
		return false
	}
}

func flattenLineOps(ops []any) []any {
	var out []any
	for _, op := range ops {
		if op == nil {
			continue
		}
		if st, ok := op.(*Stack); ok && st != nil {
			out = append(out, flattenLineOps(st.Children)...)
			continue
		}
		out = append(out, op)
	}
	return out
}

func cloneTensor[T core.Numeric](t *core.Tensor[T]) *core.Tensor[T] {
	if t == nil {
		return nil
	}
	o := core.NewTensor[T](t.Shape...)
	copy(o.Data, t.Data)
	return o
}

// TrainLineMSE injects (input, target), advances every in-flight sample one
// child, and when a sample reaches the last child, applies mode on that path.
func TrainLineMSE[T core.Numeric](ops []any, st *Line[T], input, target *core.Tensor[T], mode TrainMode, lr float64) (float64, error) {
	return trainLine(ops, st, input, target, mode, lr, mseGrad[T])
}

// TrainLineCE is TrainLineMSE with softmax-CE vs one-hot.
func TrainLineCE[T core.Numeric](ops []any, st *Line[T], input, target *core.Tensor[T], mode TrainMode, lr float64) (float64, error) {
	return trainLine(ops, st, input, target, mode, lr, ceGrad[T])
}

func trainLine[T core.Numeric](ops []any, st *Line[T], input, target *core.Tensor[T], mode TrainMode, lr float64, gap outputGap[T]) (float64, error) {
	if st == nil {
		return 0, fmt.Errorf("parallel: nil step line")
	}
	if input == nil || target == nil {
		return 0, fmt.Errorf("parallel: nil step input/target")
	}
	ops = flattenLineOps(ops)
	n := len(ops)
	if n == 0 {
		return 0, fmt.Errorf("parallel: empty step line")
	}
	st.flights = append(st.flights, lineFlight[T]{
		target: cloneTensor(target),
		ins:    make([]*core.Tensor[T], n),
		pres:   make([]*core.Tensor[T], n),
		posts:  make([]*core.Tensor[T], n),
	})
	var loss float64
	alive := st.flights[:0]
	for i := range st.flights {
		fl := &st.flights[i]
		layer := fl.stage
		if layer < 0 || layer >= n {
			continue
		}
		var in *core.Tensor[T]
		if layer == 0 {
			in = input
		} else {
			in = fl.posts[layer-1]
		}
		if in == nil {
			return 0, fmt.Errorf("parallel: step bubble at layer %d", layer)
		}
		pre, post, err := branchForward(ops[layer], in, nil)
		if err != nil {
			return 0, fmt.Errorf("parallel: step fwd layer %d: %w", layer, err)
		}
		fl.ins[layer] = cloneTensor(in)
		fl.pres[layer] = cloneTensor(pre)
		fl.posts[layer] = cloneTensor(post)
		fl.stage++
		if fl.stage < n {
			alive = append(alive, *fl)
			continue
		}
		l, gy, err := gap(fl.posts[n-1], fl.target)
		if err != nil {
			return 0, err
		}
		loss = l
		g := gy
		for j := n - 1; j >= 0; j-- {
			g, err = trainOpReturnGradIn(ops[j], g, fl.ins[j], fl.pres[j], fl.posts[j], mode, lr)
			if err != nil {
				return loss, fmt.Errorf("parallel: step bwd layer %d: %w", j, err)
			}
		}
	}
	st.flights = alive
	return loss, nil
}

func trainStackLine[T core.Numeric](s *Stack, input, target *core.Tensor[T], mode TrainMode, lr float64, gap outputGap[T]) (float64, error) {
	if s == nil {
		return 0, fmt.Errorf("parallel: nil stack")
	}
	p, ok := s.line.(*Line[T])
	if !ok || p == nil {
		p = &Line[T]{}
		s.line = p
	}
	s.SyncChildExec()
	return trainLine(s.Children, p, input, target, mode, lr, gap)
}
