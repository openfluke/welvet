package parallel

import (
	"fmt"
	"math"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/sequential"
)

// tweenLeaf is one trainable Op with the activation it actually saw.
// Whole-net tween projects the *output* gap onto this leaf's post shape
// instead of walking the Jacobian layer-by-layer.
type tweenLeaf[T core.Numeric] struct {
	op   any
	in   *core.Tensor[T]
	pre  *core.Tensor[T]
	post *core.Tensor[T]
}

// trainTweenSplitOp: one global gap, split 1/N across every trainable leaf.
// No chain rule. Stem / hemispheres / head each get a resized slice of the
// same output measurement, using that leaf's real forward input.
//
// Tried and dropped: apps/aai/test41_w_native_cam/failed.md (TweenHead, Trace).
// Keep 1/N Split.
func trainTweenSplitOp[T core.Numeric](op any, gradOut, input *core.Tensor[T], lr float64) error {
	if op == nil || input == nil {
		return fmt.Errorf("parallel: tween-split nil")
	}
	if gradOut == nil {
		return nil
	}
	var leaves []tweenLeaf[T]
	if _, err := collectTweenLeaves(op, input, &leaves); err != nil {
		return err
	}
	n := len(leaves)
	if n == 0 {
		return nil
	}
	scale := 1.0 / float64(n)
	for i, leaf := range leaves {
		if leaf.post == nil || leaf.post.Len() == 0 || !tensorFinite(leaf.post) {
			continue
		}
		gy := projectGap(gradOut, leaf.post.Shape...)
		for j := range gy.Data {
			gy.Data[j] = core.FromFloat64[T](core.AsFloat64(gy.Data[j]) * scale)
		}
		if err := applyTweenSplitLeaf(leaf, gy, lr); err != nil {
			return fmt.Errorf("tween-split leaf %d: %w", i, err)
		}
	}
	return nil
}

// applyProjectedTween is StepTween: same output gap on every leaf, resized to
// that leaf's post (so CNN/LSTM/MHA see their native rank). No 1/N split —
// that is TweenSplit. No chain rule.
func applyProjectedTween[T core.Numeric](op any, gradOut, input *core.Tensor[T], lr float64) error {
	if op == nil || input == nil || gradOut == nil {
		return nil
	}
	var leaves []tweenLeaf[T]
	if _, err := collectTweenLeaves(op, input, &leaves); err != nil {
		return err
	}
	for i, leaf := range leaves {
		if leaf.post == nil || leaf.post.Len() == 0 || !tensorFinite(leaf.post) {
			continue
		}
		gy := projectGap(gradOut, leaf.post.Shape...)
		if err := applyTweenSplitLeaf(leaf, gy, lr); err != nil {
			return fmt.Errorf("tween leaf %d: %w", i, err)
		}
	}
	return nil
}

func applyTweenSplitLeaf[T core.Numeric](leaf tweenLeaf[T], gy *core.Tensor[T], lr float64) error {
	return applyLeafDW(leaf, gy, lr, false)
}

func applyTweenSplitLeafGX[T core.Numeric](leaf tweenLeaf[T], gy *core.Tensor[T], lr float64) (*core.Tensor[T], error) {
	if gy == nil || leaf.in == nil {
		return nil, nil
	}
	pre := leaf.pre
	if pre == nil {
		p, _, err := branchForward(leaf.op, leaf.in, nil)
		if err != nil {
			return nil, err
		}
		pre = p
	}
	gx, dW, err := branchBackward(leaf.op, gy, leaf.in, nil, pre)
	if err != nil {
		return nil, err
	}
	if dW != nil && dW.Len() > 0 {
		if err := branchApplyGradSGD(leaf.op, dW, lr); err != nil {
			return gx, err
		}
	}
	return gx, nil
}

// applyLeafDW: Dense uses GradWOnly (no Wᵀ). Every other Op falls back to full
// local backward so CNN/LSTM/MHA/… still train.
func applyLeafDW[T core.Numeric](leaf tweenLeaf[T], gy *core.Tensor[T], lr float64, dwOnly bool) error {
	if gy == nil || leaf.in == nil || gy.Len() == 0 {
		return nil
	}
	if dwOnly {
		if d, ok := leaf.op.(*dense.Layer); ok {
			dW, err := dense.GradWOnly(d, gy, leaf.in, leaf.pre)
			if err != nil || dW == nil || dW.Len() == 0 {
				return nil
			}
			return dense.ApplyGradSGD(d, dW, lr)
		}
	}
	pre := leaf.pre
	if pre == nil {
		p, _, err := branchForward(leaf.op, leaf.in, nil)
		if err != nil {
			return err
		}
		pre = p
	}
	_, dW, err := branchBackward(leaf.op, gy, leaf.in, nil, pre)
	if err != nil || dW == nil || dW.Len() == 0 {
		return nil
	}
	return branchApplyGradSGD(leaf.op, dW, lr)
}

func linearGradInAny[T core.Numeric](op any, gy, input, pre *core.Tensor[T]) (gx *core.Tensor[T]) {
	defer func() {
		if rec := recover(); rec != nil {
			gx = nil
			if gy != nil && input != nil {
				gx = projectGap(gy, input.Shape...)
			}
		}
	}()
	if gy == nil {
		return nil
	}
	if d, ok := op.(*dense.Layer); ok && input != nil {
		out, err := dense.LinearGradIn(d, gy, input)
		if err == nil && out != nil {
			return out
		}
	}
	if input == nil {
		return nil
	}
	out, _, err := branchBackward(op, gy, input, nil, pre)
	if err != nil || out == nil || out.Len() == 0 {
		return projectGap(gy, input.Shape...)
	}
	return out
}

func scaleGap[T core.Numeric](gy *core.Tensor[T], scale float64) {
	if gy == nil {
		return
	}
	for j := range gy.Data {
		gy.Data[j] = core.FromFloat64[T](core.AsFloat64(gy.Data[j]) * scale)
	}
}

func trainTweenSplitFamily[T core.Numeric](op any, gradOut, input *core.Tensor[T], lr float64, mode TrainMode) error {
	if op == nil || input == nil {
		return fmt.Errorf("parallel: split family nil")
	}
	if gradOut == nil {
		return nil
	}
	var leaves []tweenLeaf[T]
	if _, err := collectTweenLeaves(op, input, &leaves); err != nil {
		return err
	}
	return trainTweenSplitLeaves(op, leaves, gradOut, lr, mode)
}

// SplitTape is one collect-walk: score Post, then Train without a second forward.
type SplitTape[T core.Numeric] struct {
	Post   *core.Tensor[T]
	leaves []tweenLeaf[T]
	host   any
}

func OpenSplitTape[T core.Numeric](op any, input *core.Tensor[T]) (*SplitTape[T], error) {
	if op == nil || input == nil {
		return nil, fmt.Errorf("parallel: OpenSplitTape nil")
	}
	var leaves []tweenLeaf[T]
	post, err := collectTweenLeaves(op, input, &leaves)
	if err != nil {
		return nil, err
	}
	return &SplitTape[T]{Post: post, leaves: leaves, host: op}, nil
}

func (t *SplitTape[T]) Train(target *core.Tensor[T], mode TrainMode, lr float64) (float64, error) {
	return t.TrainGap(target, mode, lr, mseGrad[T])
}

func (t *SplitTape[T]) TrainGap(target *core.Tensor[T], mode TrainMode, lr float64, gap outputGap[T]) (float64, error) {
	if t == nil || t.host == nil || t.Post == nil {
		return 0, fmt.Errorf("parallel: SplitTape.Train nil")
	}
	if gap == nil {
		gap = mseGrad[T]
	}
	loss, gy, err := gap(t.Post, target)
	if err != nil {
		return 0, err
	}
	return loss, trainTweenSplitLeaves(t.host, t.leaves, gy, lr, mode)
}

func trainTweenSplitLeaves[T core.Numeric](op any, leaves []tweenLeaf[T], gy *core.Tensor[T], lr float64, mode TrainMode) error {
	if gy == nil {
		return nil
	}
	switch mode.Resolve(ModeNormalBP) {
	case ModeTweenSplitHeadProxy, ModeStepTweenSplitHeadProxy:
		return trainTweenSplitHeadProxyLeaves(op, leaves, gy, lr)
	case ModeTweenSplitLinear, ModeStepTweenSplitLinear:
		return trainTweenSplitLinearLeaves(op, leaves, gy, lr, false)
	case ModeTweenSplitFastProxy, ModeMeshTweenSplitFastProxy, ModeStepTweenSplitFastProxy:
		return trainTweenSplitFastProxyLeaves(op, leaves, gy, lr)
	case ModeTweenSplitLinearCache, ModeStepTweenSplitLinearCache:
		return trainTweenSplitLinearLeaves(op, leaves, gy, lr, true)
	case ModeTweenSplitHeadProxyAsync, ModeStepTweenSplitHeadProxyAsync:
		return trainTweenSplitHeadProxyAsyncLeaves(op, leaves, gy, lr)
	case ModeTweenSplitSparse, ModeMeshTweenSplitSparse, ModeStepTweenSplitSparse:
		return trainTweenSplitSparseLeaves(op, leaves, gy, lr)
	default:
		return applySplitEven(leaves, gy, lr)
	}
}

func applySplitEven[T core.Numeric](leaves []tweenLeaf[T], gradOut *core.Tensor[T], lr float64) error {
	n := len(leaves)
	if n == 0 {
		return nil
	}
	scale := 1.0 / float64(n)
	for i, leaf := range leaves {
		if leaf.post == nil || leaf.post.Len() == 0 {
			continue
		}
		gi := projectGap(gradOut, leaf.post.Shape...)
		scaleGap(gi, scale)
		if err := applyLeafDW(leaf, gi, lr, true); err != nil {
			return fmt.Errorf("tween-split leaf %d: %w", i, err)
		}
	}
	return nil
}

func tensorFinite[T core.Numeric](t *core.Tensor[T]) bool {
	if t == nil {
		return false
	}
	for _, v := range t.Data {
		f := core.AsFloat64(v)
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return false
		}
	}
	return true
}

func collectTweenLeaves[T core.Numeric](op any, input *core.Tensor[T], leaves *[]tweenLeaf[T]) (*core.Tensor[T], error) {
	if op == nil || input == nil {
		return nil, fmt.Errorf("parallel: tween-split collect nil")
	}
	switch v := op.(type) {
	case *View:
		_, post, err := ForwardView(v, input)
		return post, err
	case *Flatten:
		_, post, err := ForwardFlatten(v, input)
		return post, err
	case *Stack:
		if v == nil {
			return nil, fmt.Errorf("parallel: nil stack")
		}
		v.SyncChildExec()
		cur := input
		for i, ch := range v.Children {
			o, err := collectTweenLeaves(ch, cur, leaves)
			if err != nil {
				return nil, fmt.Errorf("stack child %d: %w", i, err)
			}
			cur = o
		}
		return cur, nil
	case *Layer:
		if v == nil {
			return nil, fmt.Errorf("parallel: nil parallel")
		}
		v.SyncBranchExec()
		posts := make([]*core.Tensor[T], len(v.Branches))
		for i, br := range v.Branches {
			o, err := collectTweenLeaves(br, input, leaves)
			if err != nil {
				return nil, fmt.Errorf("branch %d: %w", i, err)
			}
			posts[i] = o
		}
		if v.Gate != nil {
			pre, post, err := dense.Forward(v.Gate, input)
			if err != nil {
				return nil, fmt.Errorf("gate fwd: %w", err)
			}
			*leaves = append(*leaves, tweenLeaf[T]{op: v.Gate, in: input, pre: pre, post: post})
		}
		if v.Cfg.Combine == CombineFilter {
			_, post, err := Forward(v, input)
			return post, err
		}
		return combineCollected(v, posts)
	case *sequential.Layer:
		if v == nil {
			return nil, fmt.Errorf("parallel: nil sequential")
		}
		cur := input
		for i, ch := range v.ChildOps() {
			o, err := collectTweenLeaves(ch, cur, leaves)
			if err != nil {
				return nil, fmt.Errorf("sequential child %d: %w", i, err)
			}
			cur = o
		}
		return cur, nil
	case *residual.Layer:
		if v == nil {
			return nil, fmt.Errorf("parallel: nil residual")
		}
		cur := input
		for i, ch := range v.ChildOps() {
			o, err := collectTweenLeaves(ch, cur, leaves)
			if err != nil {
				return nil, fmt.Errorf("residual child %d: %w", i, err)
			}
			cur = o
		}
		post := core.NewTensor[T](input.Shape...)
		if cur != nil && cur.Len() == input.Len() {
			for i := range post.Data {
				post.Data[i] = core.FromFloat64[T](core.AsFloat64(cur.Data[i]) + core.AsFloat64(input.Data[i]))
			}
		} else if cur != nil {
			post = cur
		}
		return post, nil
	case *ResidualSkip:
		if v == nil || v.F == nil {
			return nil, fmt.Errorf("parallel: nil ResidualSkip")
		}
		cur, err := collectTweenLeaves(v.F, input, leaves)
		if err != nil {
			return nil, err
		}
		return skipAdd(cur, input)
	default:
		pre, post, err := branchForward(op, input, nil)
		if err != nil {
			return nil, err
		}
		if branchGradWSize(op) > 0 {
			*leaves = append(*leaves, tweenLeaf[T]{op: op, in: input, pre: pre, post: post})
		}
		return post, nil
	}
}

// projectGap resizes the global output gap onto a leaf's post tensor
// (nearest / block-average). Same measurement, different width — no Jacobian.
func projectGap[T core.Numeric](src *core.Tensor[T], dstShape ...int) *core.Tensor[T] {
	dst := core.NewTensor[T](dstShape...)
	if src == nil || dst.Len() == 0 {
		return dst
	}
	sn, dn := src.Len(), dst.Len()
	if sn == dn {
		copy(dst.Data, src.Data)
		return dst
	}
	if sn <= 0 {
		return dst
	}
	for i := 0; i < dn; i++ {
		lo := i * sn / dn
		hi := (i + 1) * sn / dn
		if hi <= lo {
			hi = lo + 1
		}
		if hi > sn {
			hi = sn
		}
		var sum float64
		for j := lo; j < hi; j++ {
			sum += core.AsFloat64(src.Data[j])
		}
		dst.Data[i] = core.FromFloat64[T](sum / float64(hi-lo))
	}
	return dst
}

func combineCollected[T core.Numeric](l *Layer, branchOut []*core.Tensor[T]) (*core.Tensor[T], error) {
	if l == nil || len(branchOut) == 0 || branchOut[0] == nil {
		return nil, fmt.Errorf("parallel: combineCollected empty")
	}
	switch l.Cfg.Combine {
	case CombineConcat:
		n := branchOut[0].Shape[0]
		if len(branchOut[0].Shape) < 2 {
			n = 1
		}
		total := 0
		feats := make([]int, len(branchOut))
		for i, o := range branchOut {
			if o == nil {
				return nil, fmt.Errorf("parallel: nil branch post %d", i)
			}
			f := o.Len() / n
			if f < 1 {
				f = o.Len()
			}
			feats[i] = f
			total += f
		}
		out := core.NewTensor[T](n, total)
		for r := 0; r < n; r++ {
			off := r * total
			for i, o := range branchOut {
				f := feats[i]
				src := r * f
				copy(out.Data[off:off+f], o.Data[src:src+f])
				off += f
			}
		}
		return out, nil
	default:
		out := core.NewTensor[T](branchOut[0].Shape...)
		nb := 0
		for _, o := range branchOut {
			if o == nil || o.Len() != out.Len() {
				continue
			}
			nb++
			for j := range out.Data {
				out.Data[j] += o.Data[j]
			}
		}
		if l.Cfg.Combine == CombineAvg && nb > 0 {
			inv := 1.0 / float64(nb)
			for j := range out.Data {
				out.Data[j] = core.FromFloat64[T](core.AsFloat64(out.Data[j]) * inv)
			}
		}
		return out, nil
	}
}
