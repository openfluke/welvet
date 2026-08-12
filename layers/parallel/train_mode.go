package parallel

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
)

// TrainMode selects the update primitive for a cameral Op (or one hemisphere).
// Mirrors test41 / sine-ada perm training styles:
//
//	NormalBP / StepBP / MeshBP     — backprop + SGD (FamilyBP)
//	Tween / StepTween / MeshTween  — local / gap tween, no chain rule (FamilyTween)
//	TweenChain / StepTweenChain / MeshTweenChain — chain-rule updates (FamilyTweenChain)
//	Inherit                        — use the parent mode (BranchModes / ChildModes only)
//
// Step* vs Normal* is scheduling (online vs batched); Mesh* is volumetric forward.
// On Stack/Parallel BranchModes (no Grid), Step/Mesh collapse to their Family update.
type TrainMode uint8

const (
	ModeInherit TrainMode = iota
	ModeNormalBP
	ModeStepBP
	ModeTween
	ModeTweenChain
	ModeStepTween
	ModeStepTweenChain
	ModeMeshBP
	ModeMeshTween
	ModeMeshTweenChain
)

// ModeSGD is the legacy alias for ModeNormalBP.
const ModeSGD = ModeNormalBP

// trainFamily is the Op-level update class BranchModes dispatch on.
type trainFamily uint8

const (
	familyBP trainFamily = iota
	familyTween
	familyTweenChain
)

// AllConcreteTrainModes is the test41 set (no Inherit).
func AllConcreteTrainModes() []TrainMode {
	return []TrainMode{
		ModeNormalBP, ModeStepBP,
		ModeTween, ModeTweenChain,
		ModeStepTween, ModeStepTweenChain,
		ModeMeshBP, ModeMeshTween, ModeMeshTweenChain,
	}
}

func (m TrainMode) String() string {
	switch m {
	case ModeInherit:
		return "inherit"
	case ModeNormalBP:
		return "NormalBP"
	case ModeStepBP:
		return "StepBP"
	case ModeTween:
		return "Tween"
	case ModeTweenChain:
		return "TweenChain"
	case ModeStepTween:
		return "StepTween"
	case ModeStepTweenChain:
		return "StepTweenChain"
	case ModeMeshBP:
		return "MeshBP"
	case ModeMeshTween:
		return "MeshTween"
	case ModeMeshTweenChain:
		return "MeshTweenChain"
	default:
		return fmt.Sprintf("TrainMode(%d)", m)
	}
}

// Family collapses Step/Mesh scheduling variants to the Op update class.
func (m TrainMode) Family() trainFamily {
	switch m {
	case ModeTween, ModeStepTween, ModeMeshTween:
		return familyTween
	case ModeTweenChain, ModeStepTweenChain, ModeMeshTweenChain:
		return familyTweenChain
	default:
		return familyBP
	}
}

// RequiresGrid reports Mesh* modes that need volumetric placement for a true mesh tick.
func (m TrainMode) RequiresGrid() bool {
	switch m {
	case ModeMeshBP, ModeMeshTween, ModeMeshTweenChain:
		return true
	default:
		return false
	}
}

// UseChainRule matches test41 tween chain-rule modes.
func (m TrainMode) UseChainRule() bool {
	switch m {
	case ModeTweenChain, ModeStepTweenChain, ModeMeshTweenChain:
		return true
	default:
		return false
	}
}

// Resolve returns m unless Inherit, then parent.
func (m TrainMode) Resolve(parent TrainMode) TrainMode {
	if m == ModeInherit {
		if parent == ModeInherit {
			return ModeNormalBP
		}
		return parent
	}
	return m
}

// EffectiveBranchMode returns the train mode for branch i (Inherit → parent).
func (l *Layer) EffectiveBranchMode(i int, parent TrainMode) TrainMode {
	if l == nil || i < 0 || i >= len(l.Branches) {
		return parent.Resolve(ModeNormalBP)
	}
	if i < len(l.BranchModes) {
		return l.BranchModes[i].Resolve(parent)
	}
	return parent.Resolve(ModeNormalBP)
}

// EffectiveChildMode returns the train mode for stack child i.
func (s *Stack) EffectiveChildMode(i int, parent TrainMode) TrainMode {
	if s == nil || i < 0 || i >= len(s.Children) {
		return parent.Resolve(ModeNormalBP)
	}
	if i < len(s.ChildModes) {
		return s.ChildModes[i].Resolve(parent)
	}
	return parent.Resolve(ModeNormalBP)
}

// SetBranchModes stamps per-hemisphere train modes (len may be < Branches; rest inherit).
func (l *Layer) SetBranchModes(modes ...TrainMode) {
	if l == nil {
		return
	}
	l.BranchModes = append([]TrainMode(nil), modes...)
}

// SetChildModes stamps per-child train modes on a Stack.
func (s *Stack) SetChildModes(modes ...TrainMode) {
	if s == nil {
		return
	}
	s.ChildModes = append([]TrainMode(nil), modes...)
}

// Train applies one update to a Parallel cell under parentMode.
// gradOut is ∂L/∂y (or a tween gap); input/pre match Forward.
// When BranchModes are set, hemispheres may mix any concrete TrainModes.
func Train[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T], parentMode TrainMode, lr float64) error {
	if l == nil {
		return fmt.Errorf("parallel: Train nil")
	}
	mode := parentMode.Resolve(ModeNormalBP)
	fam := mode.Family()
	if !hasMixedBranchModes(l) && (fam == familyBP || fam == familyTweenChain) {
		_, dW, err := Backward(l, gradOut, input, pre)
		if err != nil {
			return err
		}
		return ApplyGradSGD(l, dW, lr)
	}
	return trainParallelMixed(l, gradOut, input, pre, mode, lr)
}

func hasMixedBranchModes(l *Layer) bool {
	if l == nil || len(l.BranchModes) == 0 {
		return false
	}
	for _, m := range l.BranchModes {
		if m != ModeInherit {
			return true
		}
	}
	return false
}

func trainParallelMixed[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T], parentMode TrainMode, lr float64) error {
	_ = pre
	l.SyncBranchExec()
	nb := len(l.Branches)
	if nb == 0 {
		return nil
	}
	// Recompute branch posts (same as Backward).
	branchPres := make([]*core.Tensor[T], nb)
	branchOuts := make([]*core.Tensor[T], nb)
	for i, ch := range l.Branches {
		p, o, err := branchForward(ch, input, nil)
		if err != nil {
			return fmt.Errorf("train branch %d fwd: %w", i, err)
		}
		branchPres[i], branchOuts[i] = p, o
	}
	branchGrads, err := splitCombineGrad(l, gradOut, branchOuts)
	if err != nil {
		return err
	}
	for i, ch := range l.Branches {
		bm := l.EffectiveBranchMode(i, parentMode)
		if err := trainOp(ch, branchGrads[i], input, branchPres[i], branchOuts[i], bm, lr); err != nil {
			return fmt.Errorf("train branch %d (%s): %w", i, bm, err)
		}
	}
	if l.Gate != nil {
		pf := parentMode.Resolve(ModeNormalBP).Family()
		// Gate follows BP/TweenChain parents; soft-skip on pure tween parent.
		if pf != familyTween {
			_, dWg, err := dense.Backward(l.Gate, approxGateGrad(l, gradOut, branchOuts), input, nil)
			if err == nil && dWg != nil {
				_ = dense.ApplyGradSGD(l.Gate, dWg, lr)
			}
		}
	}
	return nil
}

// splitCombineGrad maps combined ∂L/∂y into per-branch grads (add/avg/concat).
func splitCombineGrad[T core.Numeric](l *Layer, gradOut *core.Tensor[T], branchOuts []*core.Tensor[T]) ([]*core.Tensor[T], error) {
	nb := len(branchOuts)
	out := make([]*core.Tensor[T], nb)
	if gradOut == nil {
		for i := range out {
			out[i] = core.NewTensor[T](branchOuts[i].Shape...)
		}
		return out, nil
	}
	switch l.Cfg.Combine {
	case CombineAdd:
		for i := range out {
			g := core.NewTensor[T](gradOut.Shape...)
			copy(g.Data, gradOut.Data)
			out[i] = g
		}
	case CombineAvg:
		inv := core.FromFloat64[T](1.0 / float64(nb))
		for i := range out {
			g := core.NewTensor[T](gradOut.Shape...)
			for j := range g.Data {
				g.Data[j] = gradOut.Data[j] * inv
			}
			out[i] = g
		}
	case CombineConcat:
		// Slice gradOut along feature axis per branch width.
		if len(gradOut.Shape) < 2 {
			return nil, fmt.Errorf("parallel: concat grad needs rank≥2")
		}
		batch := gradOut.Shape[0]
		off := 0
		for i := 0; i < nb; i++ {
			feat := branchOuts[i].Shape[len(branchOuts[i].Shape)-1]
			g := core.NewTensor[T](batch, feat)
			for b := 0; b < batch; b++ {
				src := b*gradOut.Shape[1] + off
				dst := b * feat
				copy(g.Data[dst:dst+feat], gradOut.Data[src:src+feat])
			}
			out[i] = g
			off += feat
		}
	default: // filter / unknown — treat like add
		for i := range out {
			g := core.NewTensor[T](gradOut.Shape...)
			copy(g.Data, gradOut.Data)
			out[i] = g
		}
	}
	return out, nil
}

func approxGateGrad[T core.Numeric](l *Layer, gradOut *core.Tensor[T], branchOuts []*core.Tensor[T]) *core.Tensor[T] {
	// Soft placeholder: mean gap broadcast to gate logits [batch, branches].
	nb := len(l.Branches)
	batch := 1
	if gradOut != nil && len(gradOut.Shape) > 0 {
		batch = gradOut.Shape[0]
	}
	g := core.NewTensor[T](batch, nb)
	if gradOut == nil {
		return g
	}
	var mean float64
	for _, v := range gradOut.Data {
		mean += core.AsFloat64(v)
	}
	if len(gradOut.Data) > 0 {
		mean /= float64(len(gradOut.Data))
	}
	for i := range g.Data {
		g.Data[i] = core.FromFloat64[T](mean * 0.1)
	}
	_ = branchOuts
	return g
}

// TrainStack applies one update walking Stack children under parentMode.
// ChildModes may override per layer (including nested Parallel cameral).
// Nested Parallel BranchModes also force the mixed walker so per-hemisphere
// SGD∥Tween∥TweenChain is not silently collapsed into unified SGD.
func TrainStack[T core.Numeric](s *Stack, gradOut, input, pre *core.Tensor[T], parentMode TrainMode, lr float64) error {
	if s == nil {
		return fmt.Errorf("parallel: TrainStack nil")
	}
	mode := parentMode.Resolve(ModeNormalBP)
	fam := mode.Family()
	if !needsMixedStackTrain(s) && (fam == familyBP || fam == familyTweenChain) {
		_, dW, err := BackwardStack(s, gradOut, input, pre)
		if err != nil {
			return err
		}
		return ApplyGradSGDStack(s, dW, lr)
	}
	return trainStackMixed(s, gradOut, input, parentMode, lr)
}

func hasMixedChildModes(s *Stack) bool {
	if s == nil || len(s.ChildModes) == 0 {
		return false
	}
	for _, m := range s.ChildModes {
		if m != ModeInherit {
			return true
		}
	}
	return false
}

func needsMixedStackTrain(s *Stack) bool {
	if hasMixedChildModes(s) {
		return true
	}
	return stackHasNestedMixedModes(s)
}

func stackHasNestedMixedModes(s *Stack) bool {
	if s == nil {
		return false
	}
	for _, ch := range s.Children {
		switch v := ch.(type) {
		case *Layer:
			if hasMixedBranchModes(v) {
				return true
			}
		case *Stack:
			if needsMixedStackTrain(v) {
				return true
			}
		}
	}
	return false
}

func trainStackMixed[T core.Numeric](s *Stack, gradOut, input *core.Tensor[T], parentMode TrainMode, lr float64) error {
	s.SyncChildExec()
	n := len(s.Children)
	if n == 0 {
		return nil
	}
	ins := make([]*core.Tensor[T], n)
	pres := make([]*core.Tensor[T], n)
	posts := make([]*core.Tensor[T], n)
	current := input
	for i, ch := range s.Children {
		ins[i] = current
		p, o, err := branchForward(ch, current, nil)
		if err != nil {
			return fmt.Errorf("train stack recompute %d: %w", i, err)
		}
		pres[i], posts[i] = p, o
		current = o
	}
	gy := gradOut
	for i := n - 1; i >= 0; i-- {
		cm := s.EffectiveChildMode(i, parentMode)
		gIn, err := trainOpReturnGradIn(s.Children[i], gy, ins[i], pres[i], posts[i], cm, lr)
		if err != nil {
			return fmt.Errorf("train stack child %d (%s): %w", i, cm, err)
		}
		gy = gIn
	}
	return nil
}

func trainOp[T core.Numeric](op any, gradOut, input, pre, post *core.Tensor[T], mode TrainMode, lr float64) error {
	_, err := trainOpReturnGradIn(op, gradOut, input, pre, post, mode, lr)
	return err
}

func trainOpReturnGradIn[T core.Numeric](op any, gradOut, input, pre, post *core.Tensor[T], mode TrainMode, lr float64) (*core.Tensor[T], error) {
	mode = mode.Resolve(ModeNormalBP)
	switch mode.Family() {
	case familyTween:
		return trainTweenLocal(op, gradOut, input, post, lr)
	case familyBP, familyTweenChain:
		switch v := op.(type) {
		case *Layer:
			if hasMixedBranchModes(v) {
				if err := trainParallelMixed(v, gradOut, input, pre, mode, lr); err != nil {
					return nil, err
				}
				gIn, _, err := Backward(v, gradOut, input, pre)
				return gIn, err
			}
			gIn, dW, err := Backward(v, gradOut, input, pre)
			if err != nil {
				return nil, err
			}
			if err := ApplyGradSGD(v, dW, lr); err != nil {
				return nil, err
			}
			return gIn, nil
		case *Stack:
			if hasMixedChildModes(v) {
				if err := trainStackMixed(v, gradOut, input, mode, lr); err != nil {
					return nil, err
				}
				gIn, _, err := BackwardStack(v, gradOut, input, pre)
				return gIn, err
			}
			gIn, dW, err := BackwardStack(v, gradOut, input, pre)
			if err != nil {
				return nil, err
			}
			if err := ApplyGradSGDStack(v, dW, lr); err != nil {
				return nil, err
			}
			return gIn, nil
		default:
			if pre == nil {
				var err error
				pre, _, err = branchForward(op, input, nil)
				if err != nil {
					return nil, err
				}
			}
			gIn, dW, err := branchBackward(op, gradOut, input, nil, pre)
			if err != nil {
				return nil, err
			}
			if dW != nil && dW.Len() > 0 {
				if err := branchApplyGradSGD(op, dW, lr); err != nil {
					return nil, err
				}
			}
			return gIn, nil
		}
	default:
		return nil, fmt.Errorf("parallel: unknown train mode %s", mode)
	}
}

// trainTweenLocal: gap-style update — treat gradOut as a soft target gap on Dense leaves.
func trainTweenLocal[T core.Numeric](op any, gradOut, input, post *core.Tensor[T], lr float64) (*core.Tensor[T], error) {
	// Soft nudge: Hebbian-like via Backward with scaled gap, then half LR.
	softLR := lr * 0.5
	switch v := op.(type) {
	case *Layer:
		// Update each branch with ModeTween so BranchModes still apply; default all tween.
		return trainTweenParallel(v, gradOut, input, softLR)
	case *Stack:
		for i := len(v.Children) - 1; i >= 0; i-- {
			cm := v.EffectiveChildMode(i, ModeTween)
			if cm.Family() != familyTween {
				gIn, err := trainOpReturnGradIn(v.Children[i], gradOut, input, nil, post, cm, lr)
				if err != nil {
					return nil, err
				}
				gradOut = gIn
				continue
			}
			if err := applyTweenStores(v.Children[i], gradOut, input, softLR); err != nil {
				return nil, err
			}
		}
		gIn := core.NewTensor[T](input.Shape...)
		if gradOut != nil && gIn.Len() == gradOut.Len() {
			copy(gIn.Data, gradOut.Data)
		}
		return gIn, nil
	default:
		if err := applyTweenStores(op, gradOut, input, softLR); err != nil {
			return nil, err
		}
		gIn := core.NewTensor[T](input.Shape...)
		if gradOut != nil && gIn.Len() == gradOut.Len() {
			copy(gIn.Data, gradOut.Data)
		}
		return gIn, nil
	}
}

func trainTweenParallel[T core.Numeric](l *Layer, gradOut, input *core.Tensor[T], lr float64) (*core.Tensor[T], error) {
	pres := make([]*core.Tensor[T], len(l.Branches))
	outs := make([]*core.Tensor[T], len(l.Branches))
	for i, ch := range l.Branches {
		p, o, err := branchForward(ch, input, nil)
		if err != nil {
			return nil, err
		}
		pres[i], outs[i] = p, o
	}
	branchGrads, err := splitCombineGrad(l, gradOut, outs)
	if err != nil {
		return nil, err
	}
	for i, ch := range l.Branches {
		bm := l.EffectiveBranchMode(i, ModeTween)
		if _, err := trainOpReturnGradIn(ch, branchGrads[i], input, pres[i], outs[i], bm, lr*2); err != nil {
			return nil, err
		}
	}
	gIn := core.NewTensor[T](input.Shape...)
	if gradOut != nil && gIn.Len() == gradOut.Len() {
		copy(gIn.Data, gradOut.Data)
	}
	return gIn, nil
}

func applyTweenStores[T core.Numeric](op any, gradOut, input *core.Tensor[T], lr float64) error {
	// Soft gap update: recompute pre then Backward+SGD at soft LR.
	if dl, ok := op.(*dense.Layer); ok {
		pre, _, err := dense.Forward(dl, input)
		if err != nil {
			return err
		}
		_, dW, err := dense.Backward(dl, gradOut, input, pre)
		if err != nil {
			return err
		}
		return dense.ApplyGradSGD(dl, dW, lr)
	}
	pre, _, err := branchForward(op, input, nil)
	if err != nil {
		return err
	}
	_, dW, err := branchBackward(op, gradOut, input, nil, pre)
	if err == nil && dW != nil && dW.Len() > 0 {
		return branchApplyGradSGD(op, dW, lr)
	}
	return nil
}
