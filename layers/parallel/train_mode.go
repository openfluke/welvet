package parallel

import (
	"fmt"
	"strings"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
)

// TrainMode selects the update primitive for a cameral Op (or one hemisphere).
// Mirrors test41 / sine-ada perm training styles:
//
//	NormalBP / StepBP / MeshBP     — backprop + SGD (FamilyBP)
//	Tween / StepTween / MeshTween  — local / gap tween, no chain rule (FamilyTween)
//	TweenChain / StepTweenChain / MeshTweenChain — chain-rule updates (FamilyTweenChain)
//	TweenSplit / StepTweenSplit    — one output gap, split across every trainable leaf (FamilyTweenSplit)
//	TweenSplitHeadProxy / StepTweenSplitHeadProxy — head full J^T g_y, hidden 1/(N-1) P(gx_head)
//	TweenSplitLinear / StepTweenSplitLinear — 1/N P(W̃^T g_y), skip act' on the walk
//	TweenSplitFastProxy / StepTweenSplitFastProxy / MeshTweenSplitFastProxy
//	TweenSplitLinearCache / StepTweenSplitLinearCache
//	TweenSplitHeadProxyAsync / StepTweenSplitHeadProxyAsync
//	TweenSplitSparse / StepTweenSplitSparse / MeshTweenSplitSparse
//	TweenAlt / StepTweenAlt        — Split then Tween, repeat AltTimes (FamilyTweenAlt)
//	Freeze                         — forward/combine only; no weight update (BranchModes / ChildModes)
//	Shadow                         — frozen teacher; other cams get KD toward its post
//	Adversarial                    — train with negated LR (maximize local gap)
//	Memory                         — train only when CamKit.LastLoss ≥ SurpriseThresh
//	Inherit                        — use the parent mode (BranchModes / ChildModes only)
//
// On Stack, Step* is a 1D pipe (see Line / TrainLine). Mesh* still needs a Grid.
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
	ModeTweenSplit
	ModeStepTweenSplit
	ModeTweenAlt
	ModeStepTweenAlt
	ModeTweenSplitHeadProxy
	ModeTweenSplitLinear
	ModeTweenSplitFastProxy
	ModeTweenSplitLinearCache
	ModeTweenSplitHeadProxyAsync
	ModeTweenSplitSparse
	ModeMeshTweenSplit
	ModeMeshTweenAlt
	ModeMeshTweenSplitFastProxy
	ModeMeshTweenSplitSparse
	ModeStepTweenSplitHeadProxy
	ModeStepTweenSplitLinear
	ModeStepTweenSplitFastProxy
	ModeStepTweenSplitLinearCache
	ModeStepTweenSplitHeadProxyAsync
	ModeStepTweenSplitSparse
	ModeFreeze      // appended: keep prior iota stable for any numeric persistence
	ModeShadow      // frozen teacher + KD source
	ModeAdversarial // negated LR
	ModeMemory      // surprise-gated updates
)

// ModeSGD is the legacy alias for ModeNormalBP.
const ModeSGD = ModeNormalBP

// trainFamily is the Op-level update class BranchModes dispatch on.
type trainFamily uint8

const (
	familyBP trainFamily = iota
	familyTween
	familyTweenChain
	familyTweenSplit
	familyTweenAlt
	familyFreeze
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

// AllTrainModes is every named TrainMode including Inherit and Split/Alt credit.
func AllTrainModes() []TrainMode {
	return []TrainMode{
		ModeInherit,
		ModeNormalBP, ModeStepBP,
		ModeTween, ModeTweenChain,
		ModeStepTween, ModeStepTweenChain,
		ModeMeshBP, ModeMeshTween, ModeMeshTweenChain,
		ModeTweenSplit, ModeStepTweenSplit,
		ModeTweenAlt, ModeStepTweenAlt,
		ModeTweenSplitHeadProxy, ModeTweenSplitLinear,
		ModeTweenSplitFastProxy, ModeTweenSplitLinearCache,
		ModeTweenSplitHeadProxyAsync, ModeTweenSplitSparse,
		ModeMeshTweenSplit, ModeMeshTweenAlt,
		ModeMeshTweenSplitFastProxy, ModeMeshTweenSplitSparse,
		ModeStepTweenSplitHeadProxy, ModeStepTweenSplitLinear,
		ModeStepTweenSplitFastProxy, ModeStepTweenSplitLinearCache,
		ModeStepTweenSplitHeadProxyAsync, ModeStepTweenSplitSparse,
		ModeFreeze, ModeShadow, ModeAdversarial, ModeMemory,
	}
}

// AllCreditTrainModes is the stack-local Split / Alt credit set (scorecard §9).
// Mesh* credit is AllMeshCreditTrainModes (Needs a Grid for a true mesh tick).
func AllCreditTrainModes() []TrainMode {
	return []TrainMode{
		ModeTweenSplit, ModeStepTweenSplit,
		ModeTweenAlt, ModeStepTweenAlt,
		ModeTweenSplitHeadProxy, ModeTweenSplitLinear,
		ModeTweenSplitFastProxy, ModeTweenSplitLinearCache,
		ModeTweenSplitHeadProxyAsync, ModeTweenSplitSparse,
		ModeStepTweenSplitHeadProxy, ModeStepTweenSplitLinear,
		ModeStepTweenSplitFastProxy, ModeStepTweenSplitLinearCache,
		ModeStepTweenSplitHeadProxyAsync, ModeStepTweenSplitSparse,
	}
}

// AllMeshCreditTrainModes is Split/Alt/FastProxy/Sparse scheduled on a Grid.
func AllMeshCreditTrainModes() []TrainMode {
	return []TrainMode{
		ModeMeshTweenSplit, ModeMeshTweenAlt,
		ModeMeshTweenSplitFastProxy, ModeMeshTweenSplitSparse,
	}
}

// AllStackLocalTrainModes is every named mode TrainStackMSE can run without a Grid
// (no Inherit, no Freeze/Shadow, no Mesh*).
func AllStackLocalTrainModes() []TrainMode {
	var out []TrainMode
	for _, m := range AllTrainModes() {
		if m == ModeInherit || m.IsFrozen() || m.RequiresGrid() {
			continue
		}
		out = append(out, m)
	}
	return out
}

// AllNamedTrainModes is every concrete update: test41 nine + Split/Alt credit + Mesh*
// credit. Inherit / Freeze / Shadow are omitted (not plain weight updates).
func AllNamedTrainModes() []TrainMode {
	var out []TrainMode
	for _, m := range AllTrainModes() {
		if m == ModeInherit || m.IsFrozen() {
			continue
		}
		out = append(out, m)
	}
	return out
}

// ParseTrainMode maps a persistence / CLI name to TrainMode.
// Empty string is Inherit. Aliases: sgd/bp → NormalBP, fastproxy → TweenSplitFastProxy, sparse → TweenSplitSparse.
func ParseTrainMode(s string) (TrainMode, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return ModeInherit, nil
	}
	for _, m := range AllTrainModes() {
		if strings.EqualFold(m.String(), s) {
			return m, nil
		}
	}
	key := strings.ToLower(strings.ReplaceAll(strings.ReplaceAll(s, "_", ""), "-", ""))
	switch key {
	case "bp", "sgd", "normal":
		return ModeNormalBP, nil
	case "fastproxy", "fast":
		return ModeTweenSplitFastProxy, nil
	case "sparse":
		return ModeTweenSplitSparse, nil
	case "headproxy":
		return ModeTweenSplitHeadProxy, nil
	case "linear":
		return ModeTweenSplitLinear, nil
	case "linearcache":
		return ModeTweenSplitLinearCache, nil
	case "headproxyasync", "asyncproxy":
		return ModeTweenSplitHeadProxyAsync, nil
	case "alt":
		return ModeTweenAlt, nil
	case "split":
		return ModeTweenSplit, nil
	case "meshsplit", "meshtweensplit":
		return ModeMeshTweenSplit, nil
	case "meshfastproxy", "meshfast":
		return ModeMeshTweenSplitFastProxy, nil
	case "meshsparse":
		return ModeMeshTweenSplitSparse, nil
	case "meshalt":
		return ModeMeshTweenAlt, nil
	case "stepheadproxy", "steptweensplitheadproxy":
		return ModeStepTweenSplitHeadProxy, nil
	case "steplinear", "steptweensplitlinear":
		return ModeStepTweenSplitLinear, nil
	case "stepfastproxy", "stepfast", "steptweensplitfastproxy":
		return ModeStepTweenSplitFastProxy, nil
	case "steplinearcache", "steptweensplitlinearcache":
		return ModeStepTweenSplitLinearCache, nil
	case "stepheadproxyasync", "stepasyncproxy", "steptweensplitheadproxyasync":
		return ModeStepTweenSplitHeadProxyAsync, nil
	case "stepsparse", "steptweensplitsparse":
		return ModeStepTweenSplitSparse, nil
	case "freeze", "frozen", "notrain", "eval":
		return ModeFreeze, nil
	case "shadow", "teacher", "kd":
		return ModeShadow, nil
	case "adversarial", "adv", "enemy":
		return ModeAdversarial, nil
	case "memory", "surprise", "hippo":
		return ModeMemory, nil
	default:
		return ModeInherit, fmt.Errorf("parallel: unknown train mode %q", s)
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
	case ModeTweenSplit:
		return "TweenSplit"
	case ModeStepTweenSplit:
		return "StepTweenSplit"
	case ModeTweenAlt:
		return "TweenAlt"
	case ModeStepTweenAlt:
		return "StepTweenAlt"
	case ModeTweenSplitHeadProxy:
		return "TweenSplitHeadProxy"
	case ModeTweenSplitLinear:
		return "TweenSplitLinear"
	case ModeTweenSplitFastProxy:
		return "TweenSplitFastProxy"
	case ModeTweenSplitLinearCache:
		return "TweenSplitLinearCache"
	case ModeTweenSplitHeadProxyAsync:
		return "TweenSplitHeadProxyAsync"
	case ModeTweenSplitSparse:
		return "TweenSplitSparse"
	case ModeMeshTweenSplit:
		return "MeshTweenSplit"
	case ModeMeshTweenAlt:
		return "MeshTweenAlt"
	case ModeMeshTweenSplitFastProxy:
		return "MeshTweenSplitFastProxy"
	case ModeMeshTweenSplitSparse:
		return "MeshTweenSplitSparse"
	case ModeStepTweenSplitHeadProxy:
		return "StepTweenSplitHeadProxy"
	case ModeStepTweenSplitLinear:
		return "StepTweenSplitLinear"
	case ModeStepTweenSplitFastProxy:
		return "StepTweenSplitFastProxy"
	case ModeStepTweenSplitLinearCache:
		return "StepTweenSplitLinearCache"
	case ModeStepTweenSplitHeadProxyAsync:
		return "StepTweenSplitHeadProxyAsync"
	case ModeStepTweenSplitSparse:
		return "StepTweenSplitSparse"
	case ModeFreeze:
		return "Freeze"
	case ModeShadow:
		return "Shadow"
	case ModeAdversarial:
		return "Adversarial"
	case ModeMemory:
		return "Memory"
	default:
		return fmt.Sprintf("TrainMode(%d)", m)
	}
}

// IsFrozen is forward/combine only — no weight update (Freeze or Shadow teacher).
func (m TrainMode) IsFrozen() bool {
	return m == ModeFreeze || m == ModeShadow
}

// IsShadow marks a frozen KD teacher cam.
func (m TrainMode) IsShadow() bool {
	return m == ModeShadow
}

// Family collapses Step/Mesh scheduling variants to the Op update class.
func (m TrainMode) Family() trainFamily {
	switch m {
	case ModeFreeze, ModeShadow:
		return familyFreeze
	case ModeAdversarial, ModeMemory:
		return familyBP // update path still BP-shaped; signs/gates applied in mixed train
	case ModeTween, ModeStepTween, ModeMeshTween:
		return familyTween
	case ModeTweenChain, ModeStepTweenChain, ModeMeshTweenChain:
		return familyTweenChain
	case ModeTweenSplit, ModeStepTweenSplit, ModeMeshTweenSplit,
		ModeTweenSplitHeadProxy, ModeTweenSplitLinear,
		ModeTweenSplitFastProxy, ModeTweenSplitLinearCache,
		ModeTweenSplitHeadProxyAsync, ModeTweenSplitSparse,
		ModeMeshTweenSplitFastProxy, ModeMeshTweenSplitSparse,
		ModeStepTweenSplitHeadProxy, ModeStepTweenSplitLinear,
		ModeStepTweenSplitFastProxy, ModeStepTweenSplitLinearCache,
		ModeStepTweenSplitHeadProxyAsync, ModeStepTweenSplitSparse:
		return familyTweenSplit
	case ModeTweenAlt, ModeStepTweenAlt, ModeMeshTweenAlt:
		return familyTweenAlt
	default:
		return familyBP
	}
}

// RequiresGrid reports Mesh* modes that need volumetric placement for a true mesh tick.
func (m TrainMode) RequiresGrid() bool {
	switch m {
	case ModeMeshBP, ModeMeshTween, ModeMeshTweenChain,
		ModeMeshTweenSplit, ModeMeshTweenAlt,
		ModeMeshTweenSplitFastProxy, ModeMeshTweenSplitSparse:
		return true
	default:
		return false
	}
}

// IsSplitFamily is the one-tape Split credit class (not Alt).
func (m TrainMode) IsSplitFamily() bool {
	return m.Family() == familyTweenSplit
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
// When BranchModes are set, hemispheres may mix any concrete TrainModes
// (including Freeze: that cam still forwards/combines, but skips ApplyGrad).
func Train[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T], parentMode TrainMode, lr float64) error {
	if l == nil {
		return fmt.Errorf("parallel: Train nil")
	}
	mode := parentMode.Resolve(ModeNormalBP)
	if mode.IsFrozen() {
		return nil
	}
	fam := mode.Family()
	if fam == familyTweenSplit {
		return trainTweenSplitFamily(l, gradOut, input, lr, mode)
	}
	if fam == familyTweenAlt {
		return trainTweenAltOp(l, gradOut, input, altTimesOf(l.AltTimes), lr)
	}
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
	before := snapshotBranchNorms(l)
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
	// Shadow KD: pull trainable cams toward teacher post.
	var teacher *core.Tensor[T]
	for i := 0; i < nb; i++ {
		if l.EffectiveBranchMode(i, parentMode).IsShadow() {
			teacher = branchOuts[i]
			break
		}
	}
	if teacher != nil {
		coef := 1.0
		if l.CamKit != nil {
			coef = l.CamKit.shadowCoef()
		}
		n := float64(teacher.Len())
		if n < 1 {
			n = 1
		}
		for i := 0; i < nb; i++ {
			bm := l.EffectiveBranchMode(i, parentMode)
			if bm.IsFrozen() || branchGrads[i] == nil || branchOuts[i] == nil {
				continue
			}
			if branchOuts[i].Len() != teacher.Len() {
				continue
			}
			for j := range branchGrads[i].Data {
				diff := core.AsFloat64(branchOuts[i].Data[j]) - core.AsFloat64(teacher.Data[j])
				branchGrads[i].Data[j] += core.FromFloat64[T](coef * 2 * diff / n)
			}
		}
	}
	memOK := l.memoryAllowsTrain()
	for i, ch := range l.Branches {
		bm := l.EffectiveBranchMode(i, parentMode)
		if bm.IsFrozen() {
			continue
		}
		if bm == ModeMemory && !memOK {
			continue
		}
		blr := l.EffectiveBranchLR(i, lr)
		if blr == 0 {
			continue
		}
		if bm == ModeAdversarial {
			blr = -blr
		}
		updateMode := bm
		switch bm {
		case ModeAdversarial, ModeMemory:
			updateMode = parentMode.Resolve(ModeNormalBP)
		}
		if err := trainOp(ch, branchGrads[i], input, branchPres[i], branchOuts[i], updateMode, blr); err != nil {
			return fmt.Errorf("train branch %d (%s): %w", i, bm, err)
		}
	}
	if l.Gate != nil {
		pf := parentMode.Resolve(ModeNormalBP).Family()
		if pf != familyTween {
			gatePre, _, gErr := dense.Forward(l.Gate, input)
			if gErr == nil {
				_, dWg, err := dense.Backward(l.Gate, approxGateGrad(l, gradOut, branchOuts), input, gatePre)
				if err == nil && dWg != nil {
					_ = dense.ApplyGradSGD(l.Gate, dWg, lr)
				}
			}
		}
	}
	recordPlasticity(l, before)
	_ = l.applyDNAReg()
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
	case CombineMax:
		// Straight-through: route gy only to argmax branch per element.
		for i := range out {
			out[i] = core.NewTensor[T](gradOut.Shape...)
		}
		n := gradOut.Len()
		for j := 0; j < n; j++ {
			bestI := 0
			best := core.AsFloat64(branchOuts[0].Data[j])
			for i := 1; i < nb; i++ {
				v := core.AsFloat64(branchOuts[i].Data[j])
				if v > best {
					best, bestI = v, i
				}
			}
			out[bestI].Data[j] = gradOut.Data[j]
		}
	case CombineSparseK:
		sel := sparseKSelected(branchOuts, l.Cfg.sparseK())
		inv := core.FromFloat64[T](1.0 / float64(len(sel)))
		for i := range out {
			out[i] = core.NewTensor[T](gradOut.Shape...)
		}
		for _, i := range sel {
			for j := range out[i].Data {
				out[i].Data[j] = gradOut.Data[j] * inv
			}
		}
	case CombineDisagree:
		beta := l.Cfg.disagreeBeta()
		inv := 1.0 / float64(nb)
		for i := range out {
			out[i] = core.NewTensor[T](gradOut.Shape...)
		}
		if nb == 2 {
			// y = mean + β(a−b); ∂y/∂a = 0.5+β, ∂y/∂b = 0.5−β
			ca := core.FromFloat64[T](inv + beta)
			cb := core.FromFloat64[T](inv - beta)
			for j := range gradOut.Data {
				out[0].Data[j] = gradOut.Data[j] * ca
				out[1].Data[j] = gradOut.Data[j] * cb
			}
		} else {
			// y = mean + β(cam0−mean) = (1−β)·mean + β·cam0
			c0 := core.FromFloat64[T](beta + (1-beta)*inv)
			co := core.FromFloat64[T]((1 - beta) * inv)
			for j := range gradOut.Data {
				out[0].Data[j] = gradOut.Data[j] * c0
				for i := 1; i < nb; i++ {
					out[i].Data[j] = gradOut.Data[j] * co
				}
			}
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
	if mode.IsFrozen() {
		return nil
	}
	fam := mode.Family()
	if fam == familyTweenSplit {
		return trainTweenSplitFamily(s, gradOut, input, lr, mode)
	}
	if fam == familyTweenAlt {
		return trainTweenAltOp(s, gradOut, input, altTimesOf(s.AltTimes), lr)
	}
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
	if mode.IsFrozen() {
		return frozenGradIn(op, gradOut, input, pre, post)
	}
	switch mode.Family() {
	case familyTween:
		return trainTweenLocal(op, gradOut, input, post, lr)
	case familyTweenSplit:
		if err := trainTweenSplitFamily(op, gradOut, input, lr, mode); err != nil {
			return nil, err
		}
		gIn := core.NewTensor[T](input.Shape...)
		if gradOut != nil && gIn.Len() == gradOut.Len() {
			copy(gIn.Data, gradOut.Data)
		}
		return gIn, nil
	case familyTweenAlt:
		times := 1
		switch v := op.(type) {
		case *Stack:
			times = altTimesOf(v.AltTimes)
		case *Layer:
			times = altTimesOf(v.AltTimes)
		}
		if err := trainTweenAltOp(op, gradOut, input, times, lr); err != nil {
			return nil, err
		}
		gIn := core.NewTensor[T](input.Shape...)
		if gradOut != nil && gIn.Len() == gradOut.Len() {
			copy(gIn.Data, gradOut.Data)
		}
		return gIn, nil
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

// frozenGradIn passes ∂L/∂x through without ApplyGrad (Freeze BranchMode / ChildMode).
func frozenGradIn[T core.Numeric](op any, gradOut, input, pre, post *core.Tensor[T]) (*core.Tensor[T], error) {
	_ = post
	switch v := op.(type) {
	case *Layer:
		gIn, _, err := Backward(v, gradOut, input, pre)
		return gIn, err
	case *Stack:
		gIn, _, err := BackwardStack(v, gradOut, input, pre)
		return gIn, err
	default:
		if pre == nil {
			var err error
			pre, _, err = branchForward(op, input, nil)
			if err != nil {
				return nil, err
			}
		}
		gIn, _, err := branchBackward(op, gradOut, input, nil, pre)
		return gIn, err
	}
}

// trainTweenLocal: no chain rule. Project the output gap onto every trainable
// leaf's real post shape (CNN/LSTM/MHA included) using that leaf's forward input.
func trainTweenLocal[T core.Numeric](op any, gradOut, input, post *core.Tensor[T], lr float64) (*core.Tensor[T], error) {
	_ = post
	if input == nil {
		return nil, fmt.Errorf("parallel: tween nil input")
	}
	softLR := lr * 0.5
	if err := applyProjectedTween(op, gradOut, input, softLR); err != nil {
		return nil, err
	}
	gIn := core.NewTensor[T](input.Shape...)
	if gradOut != nil && gIn.Len() == gradOut.Len() {
		copy(gIn.Data, gradOut.Data)
	}
	return gIn, nil
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
		if bm.IsFrozen() {
			continue
		}
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
