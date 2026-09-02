package parallel

import (
	"fmt"
	"time"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/cnn2"
	"github.com/openfluke/welvet/layers/cnn3"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/quant"
)

// Stack is a heterogeneous sequential chain of cell Ops (Dense, Parallel,
// Sequential, nested Stack, CNN, …). It is the host for nested multi-cameral
// graphs: place Parallel (hemispheres) at any depth inside Children.
//
// Lives in package parallel so child/branch dispatch shares one type switch
// without an import cycle between stack↔parallel.
type Stack struct {
	Core       core.Layer
	Exec       core.ExecConfig
	Children   []any
	ChildModes []TrainMode // optional per-child mode; empty ⇒ inherit parent
	AltTimes   int         // TweenAlt: Split→Tween pairs per update (0 ⇒ 1)
	CamSync    *CamSyncConfig // optional inter-cameral / cross-layer weight averaging
	Tanhi      any            // optional *tanhi.UDPConfig; synced from grid on PlaceStack
	accel      splitAccel  // LinearCache / HeadProxyAsync / Sparse
	line       any         // *Line[T] for Step* systolic train
}

// NewStack builds a Stack from ordered Ops. Each child keeps its own
// dtype/format storage truth.
func NewStack(children ...any) (*Stack, error) {
	if len(children) == 0 {
		return nil, fmt.Errorf("parallel: NewStack needs ≥1 child")
	}
	for i, ch := range children {
		if ch == nil {
			return nil, fmt.Errorf("parallel: nil stack child at %d", i)
		}
	}
	inH := opInputHeight(children[0])
	outH := opOutputHeight(children[len(children)-1])
	s := &Stack{
		Core: core.Layer{
			Type:         core.LayerStack,
			DType:        core.DTypeFloat32,
			Activation:   core.ActivationLinear,
			InputHeight:  inH,
			OutputHeight: outH,
			TileSize:     32,
			MultiCore:    true,
		},
		Children: append([]any(nil), children...),
		Exec: core.ExecConfig{
			Backend:   core.BackendCPUTiled,
			MultiCore: true,
			TileSize:  32,
		},
	}
	s.SyncChildExec()
	return s, nil
}

func opInputHeight(op any) int {
	if v, ok := op.(*View); ok && v != nil && len(v.Shape) > 0 {
		return v.Shape[len(v.Shape)-1]
	}
	switch v := op.(type) {
	case *dense.Layer:
		return v.Core.InputHeight
	case *Layer:
		return v.Core.InputHeight
	case *Stack:
		return v.Core.InputHeight
	case *sequential.Layer:
		return v.Core.InputHeight
	case *residual.Layer:
		return v.Core.InputHeight
	case *ResidualSkip:
		return opInputHeight(v.F)
	case *cnn1.Layer:
		return v.Core.InputHeight
	case *cnn2.Layer:
		return v.Core.InputHeight
	case *cnn3.Layer:
		return v.Core.InputHeight
	default:
		return 0
	}
}

func opOutputHeight(op any) int {
	if v, ok := op.(*View); ok && v != nil && len(v.Shape) > 0 {
		return v.Shape[len(v.Shape)-1]
	}
	switch v := op.(type) {
	case *dense.Layer:
		return v.Core.OutputHeight
	case *Layer:
		return v.Core.OutputHeight
	case *Stack:
		return v.Core.OutputHeight
	case *sequential.Layer:
		return v.Core.OutputHeight
	case *residual.Layer:
		return v.Core.OutputHeight
	case *ResidualSkip:
		return opOutputHeight(v.F)
	case *cnn1.Layer:
		return v.Core.OutputHeight
	case *cnn2.Layer:
		return v.Core.OutputHeight
	case *cnn3.Layer:
		return v.Core.OutputHeight
	default:
		return 0
	}
}

// SyncChildExec copies Exec onto every child Op.
func (s *Stack) SyncChildExec() {
	if s == nil {
		return
	}
	for _, ch := range s.Children {
		branchSyncExec(ch, s.Exec)
	}
}

// SetDType sets dtype on every child Op.
func (s *Stack) SetDType(dt core.DType) error {
	if s == nil {
		return fmt.Errorf("parallel: nil stack")
	}
	for i, ch := range s.Children {
		if err := branchSetDType(ch, dt); err != nil {
			return fmt.Errorf("stack child %d: %w", i, err)
		}
	}
	s.Core.DType = dt
	return nil
}

// Pack packs every child Op.
func (s *Stack) Pack(format quant.Format) error {
	if s == nil {
		return fmt.Errorf("parallel: nil stack")
	}
	for i, ch := range s.Children {
		if err := branchPack(ch, format); err != nil {
			return fmt.Errorf("stack child %d: %w", i, err)
		}
	}
	return nil
}

// Forward chains children. Backend is whatever each child Exec says after SyncChildExec.
func ForwardStack[T core.Numeric](s *Stack, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if s == nil || input == nil {
		return nil, nil, fmt.Errorf("parallel: nil stack/input")
	}
	s.SyncChildExec()
	if len(s.Children) == 0 {
		out := core.NewTensor[T](input.Shape...)
		copy(out.Data, input.Data)
		return out, out, nil
	}
	current := input
	var lastPre *core.Tensor[T]
	stackT0 := time.Now()
	for i, ch := range s.Children {
		t0 := time.Now()
		p, o, errFwd := branchForward(ch, current, nil)
		if errFwd != nil {
			return nil, nil, fmt.Errorf("stack fwd child %d: %w", i, errFwd)
		}
		emitOpTanhi(s.Tanhi, "fwd", i, ch, t0, time.Now(), o)
		lastPre = p
		current = o
	}
	emitStackTanhi(s, "fwd", len(s.Children), stackT0, time.Now(), current)
	return lastPre, current, nil
}

// Backward recomputes the forward tape then chains branchBackward; gradW is concat of child dWs.
func BackwardStack[T core.Numeric](s *Stack, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	_ = pre
	if s == nil {
		return nil, nil, fmt.Errorf("parallel: nil stack")
	}
	s.SyncChildExec()
	if len(s.Children) == 0 {
		gi := core.NewTensor[T](input.Shape...)
		if gradOut != nil {
			copy(gi.Data, gradOut.Data)
		}
		return gi, nil, nil
	}
	n := len(s.Children)
	ins := make([]*core.Tensor[T], n)
	pres := make([]*core.Tensor[T], n)
	current := input
	for i, ch := range s.Children {
		ins[i] = current
		p, o, err := branchForward(ch, current, nil)
		if err != nil {
			return nil, nil, fmt.Errorf("stack recompute child %d: %w", i, err)
		}
		pres[i] = p
		current = o
	}
	gy := gradOut
	dWs := make([]*core.Tensor[T], n)
	stackT0 := time.Now()
	for i := n - 1; i >= 0; i-- {
		t0 := time.Now()
		gx, dw, errBwd := branchBackward(s.Children[i], gy, ins[i], nil, pres[i])
		if errBwd != nil {
			return nil, nil, fmt.Errorf("stack bwd child %d: %w", i, errBwd)
		}
		emitOpTanhi(s.Tanhi, "bwd", i, s.Children[i], t0, time.Now(), gx)
		dWs[i] = dw
		gy = gx
	}
	emitStackTanhi(s, "bwd", n, stackT0, time.Now(), gy)
	need := s.GradWSize()
	if need == 0 {
		return gy, nil, nil
	}
	gradW = core.NewTensor[T](need)
	off := 0
	for _, dw := range dWs {
		if dw == nil || dw.Len() == 0 {
			continue
		}
		copy(gradW.Data[off:], dw.Data)
		off += dw.Len()
	}
	return gy, gradW, nil
}

// GradWSize is the sum of child GradWSize.
func (s *Stack) GradWSize() int {
	if s == nil {
		return 0
	}
	n := 0
	for _, ch := range s.Children {
		n += branchGradWSize(ch)
	}
	return n
}

// ApplyGradSGDStack splits concat dW across stack children.
func ApplyGradSGDStack[T core.Numeric](s *Stack, dW *core.Tensor[T], lr float64) error {
	if s == nil {
		return fmt.Errorf("parallel: ApplyGradSGDStack nil")
	}
	if dW == nil {
		return nil
	}
	off := 0
	for i, ch := range s.Children {
		n := branchGradWSize(ch)
		if n == 0 {
			continue
		}
		if off+n > dW.Len() {
			return fmt.Errorf("stack: dW short at child %d (need %d, have %d)", i, off+n, dW.Len())
		}
		slice := core.NewTensor[T](n)
		copy(slice.Data, dW.Data[off:off+n])
		if err := branchApplyGradSGD(ch, slice, lr); err != nil {
			return fmt.Errorf("stack child %d SGD: %w", i, err)
		}
		off += n
	}
	return nil
}
