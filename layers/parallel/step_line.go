package parallel

import (
	"fmt"

	"github.com/openfluke/welvet/core"
)

// Line is a systolic 1D pipe: one new sample at layer 0 per Tick, every
// in-flight sample advances one child. Output at depth D is the sample that
// entered D ticks ago. Train when a RoleTrain sample pops the last child,
// using the activations saved along its path (not a full-chain re-forward of
// the latest x). RoleAction samples pop as actionable throughput only.
type Line[T core.Numeric] struct {
	flights []lineFlight[T]
}

// FlightRole tags why a sample is in the systolic pipe.
type FlightRole uint8

const (
	// RoleTrain applies credit when the sample pops the output.
	RoleTrain FlightRole = iota
	// RoleAction yields ActionOut on pop — no credit update.
	RoleAction
)

type lineFlight[T core.Numeric] struct {
	role             FlightRole
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

// InFlight returns how many samples are currently in the pipe.
func (st *Line[T]) InFlight() int {
	if st == nil {
		return 0
	}
	return len(st.flights)
}

// TrainLineMSE injects a RoleTrain sample, advances every in-flight sample one
// child, and when a train sample reaches the last child, applies mode on that path.
func TrainLineMSE[T core.Numeric](ops []any, st *Line[T], input, target *core.Tensor[T], mode TrainMode, lr float64) (float64, error) {
	loss, _, err := TickLine(ops, st, input, target, RoleTrain, mode, lr, mseGrad[T])
	return loss, err
}

// TrainLineCE is TrainLineMSE with softmax-CE vs one-hot.
func TrainLineCE[T core.Numeric](ops []any, st *Line[T], input, target *core.Tensor[T], mode TrainMode, lr float64) (float64, error) {
	loss, _, err := TickLine(ops, st, input, target, RoleTrain, mode, lr, ceGrad[T])
	return loss, err
}

// TickLine injects one sample (train or action) and advances every in-flight
// sample one child. RoleTrain requires target and may return loss on pop.
// RoleAction ignores target and may return actionOut on pop (no credit).
func TickLine[T core.Numeric](
	ops []any,
	st *Line[T],
	input, target *core.Tensor[T],
	role FlightRole,
	mode TrainMode,
	lr float64,
	gap outputGap[T],
) (loss float64, actionOut *core.Tensor[T], err error) {
	if gap == nil {
		gap = mseGrad[T]
	}
	if st == nil {
		return 0, nil, fmt.Errorf("parallel: nil step line")
	}
	if input == nil {
		return 0, nil, fmt.Errorf("parallel: nil step input")
	}
	if role == RoleTrain && target == nil {
		return 0, nil, fmt.Errorf("parallel: RoleTrain needs target")
	}
	ops = flattenLineOps(ops)
	n := len(ops)
	if n == 0 {
		return 0, nil, fmt.Errorf("parallel: empty step line")
	}
	flNew := lineFlight[T]{
		role:  role,
		ins:   make([]*core.Tensor[T], n),
		pres:  make([]*core.Tensor[T], n),
		posts: make([]*core.Tensor[T], n),
	}
	if role == RoleTrain {
		flNew.target = cloneTensor(target)
	}
	st.flights = append(st.flights, flNew)

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
			return 0, nil, fmt.Errorf("parallel: step bubble at layer %d", layer)
		}
		pre, post, errFwd := branchForward(ops[layer], in, nil)
		if errFwd != nil {
			return 0, nil, fmt.Errorf("parallel: step fwd layer %d: %w", layer, errFwd)
		}
		fl.ins[layer] = cloneTensor(in)
		fl.pres[layer] = cloneTensor(pre)
		fl.posts[layer] = cloneTensor(post)
		fl.stage++
		if fl.stage < n {
			alive = append(alive, *fl)
			continue
		}
		// Popped output.
		if fl.role == RoleAction {
			actionOut = cloneTensor(fl.posts[n-1])
			continue
		}
		l, gy, errGap := gap(fl.posts[n-1], fl.target)
		if errGap != nil {
			return 0, actionOut, errGap
		}
		loss = l
		g := gy
		for j := n - 1; j >= 0; j-- {
			g, err = trainOpReturnGradIn(ops[j], g, fl.ins[j], fl.pres[j], fl.posts[j], mode, lr)
			if err != nil {
				return loss, actionOut, fmt.Errorf("parallel: step bwd layer %d: %w", j, err)
			}
		}
	}
	st.flights = alive
	return loss, actionOut, nil
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
	loss, _, err := TickLine(s.Children, p, input, target, RoleTrain, mode, lr, gap)
	return loss, err
}

// TickStackLine is the Stack entry for train/action systolic ticks.
func TickStackLine[T core.Numeric](
	s *Stack,
	input, target *core.Tensor[T],
	role FlightRole,
	mode TrainMode,
	lr float64,
) (loss float64, actionOut *core.Tensor[T], err error) {
	if s == nil {
		return 0, nil, fmt.Errorf("parallel: nil stack")
	}
	p, ok := s.line.(*Line[T])
	if !ok || p == nil {
		p = &Line[T]{}
		s.line = p
	}
	s.SyncChildExec()
	loss, actionOut, err = TickLine(s.Children, p, input, target, role, mode, lr, mseGrad[T])
	if err != nil {
		return loss, actionOut, err
	}
	_ = s.MaybeSync(SyncAfterStep)
	_ = s.MaybeSync(SyncAfterSample)
	return loss, actionOut, nil
}

// StackLineDepth is the flattened hop count used by the systolic pipe.
func StackLineDepth(s *Stack) int {
	if s == nil {
		return 0
	}
	return len(flattenLineOps(s.Children))
}

// StackInFlight returns queued samples in the Stack's Line (0 if unused).
func StackInFlight(s *Stack) int {
	if s == nil {
		return 0
	}
	p, ok := s.line.(*Line[float32])
	if !ok || p == nil {
		return 0
	}
	return p.InFlight()
}
