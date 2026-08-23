package parallel

import (
	"math"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/sequential"
)

const defaultLinearCacheEvery = 20

type splitAccel struct {
	proxy     []float64
	cache     [][]float64
	cacheNorm float64
	tick      int
	// CacheEvery: LinearCache refresh period (0 ⇒ 20).
	CacheEvery int
	sparse     int
}

func (a *splitAccel) every() int {
	if a == nil || a.CacheEvery <= 0 {
		return defaultLinearCacheEvery
	}
	return a.CacheEvery
}

func accelOf(op any) *splitAccel {
	switch v := op.(type) {
	case *Stack:
		return &v.accel
	case *Layer:
		return &v.accel
	default:
		return nil
	}
}

// ClearSplitAccel drops LinearCache / HeadProxy / Sparse scratch so a NaN
// recovery (or weight reinject) does not keep feeding poison into the next step.
func (s *Stack) ClearSplitAccel() {
	if s == nil {
		return
	}
	s.accel = splitAccel{CacheEvery: s.accel.CacheEvery}
}

func (l *Layer) ClearSplitAccel() {
	if l == nil {
		return
	}
	l.accel = splitAccel{CacheEvery: l.accel.CacheEvery}
}

func trainTweenSplitHeadProxyLeaves[T core.Numeric](_ any, leaves []tweenLeaf[T], gy *core.Tensor[T], lr float64) error {
	n := len(leaves)
	if n == 0 {
		return nil
	}
	head := leaves[n-1]
	if head.post == nil || head.post.Len() == 0 {
		return applySplitEven(leaves, gy, lr)
	}
	gHead := projectGap(gy, head.post.Shape...)
	gx, err := applyTweenSplitLeafGX(head, gHead, lr)
	if err != nil {
		return err
	}
	if n == 1 {
		return nil
	}
	src := gx
	if src == nil || src.Len() == 0 {
		src = gy
	}
	return applyHiddenDW(leaves[:n-1], src, 1.0/float64(n-1), lr)
}

func applyHiddenDW[T core.Numeric](leaves []tweenLeaf[T], src *core.Tensor[T], scale, lr float64) error {
	for _, leaf := range leaves {
		if leaf.post == nil || leaf.post.Len() == 0 {
			continue
		}
		gi := projectGap(src, leaf.post.Shape...)
		scaleGap(gi, scale)
		if err := applyLeafDW(leaf, gi, lr, true); err != nil {
			return err
		}
	}
	return nil
}

func trainTweenSplitFastProxyLeaves[T core.Numeric](_ any, leaves []tweenLeaf[T], gy *core.Tensor[T], lr float64) error {
	n := len(leaves)
	if n == 0 {
		return nil
	}
	head := leaves[n-1]
	proxy := linearGradInAny(head.op, gy, head.in, nil)
	if head.post != nil && head.post.Len() > 0 {
		if err := applyLeafDW(head, projectGap(gy, head.post.Shape...), lr, true); err != nil {
			return err
		}
	}
	if n == 1 {
		return nil
	}
	src := proxy
	if src == nil || src.Len() == 0 {
		src = gy
	}
	return applyHiddenDW(leaves[:n-1], src, 1.0/float64(n-1), lr)
}

func trainTweenSplitHeadProxyAsyncLeaves[T core.Numeric](op any, leaves []tweenLeaf[T], gy *core.Tensor[T], lr float64) error {
	n := len(leaves)
	if n == 0 {
		return nil
	}
	acc := accelOf(op)
	head := leaves[n-1]
	appliedHidden := false
	if n > 1 && acc != nil && len(acc.proxy) > 0 {
		stale := core.NewTensor[T](len(acc.proxy))
		for i := range acc.proxy {
			stale.Data[i] = core.FromFloat64[T](acc.proxy[i])
		}
		if err := applyHiddenDW(leaves[:n-1], stale, 1.0/float64(n-1), lr); err != nil {
			return err
		}
		appliedHidden = true
	}
	gHead := gy
	if head.post != nil && head.post.Len() > 0 {
		gHead = projectGap(gy, head.post.Shape...)
	}
	if err := applyLeafDW(head, gHead, lr, true); err != nil {
		return err
	}
	proxy := linearGradInAny(head.op, gy, head.in, nil)
	if acc != nil && proxy != nil && proxy.Len() > 0 {
		acc.proxy = make([]float64, proxy.Len())
		for i := range proxy.Data {
			acc.proxy[i] = core.AsFloat64(proxy.Data[i])
		}
	}
	if n > 1 && !appliedHidden {
		src := proxy
		if src == nil {
			src = gy
		}
		return applyHiddenDW(leaves[:n-1], src, 1.0/float64(n-1), lr)
	}
	return nil
}

func trainTweenSplitSparseLeaves[T core.Numeric](op any, leaves []tweenLeaf[T], gy *core.Tensor[T], lr float64) error {
	n := len(leaves)
	if n == 0 {
		return nil
	}
	head := leaves[n-1]
	proxy := linearGradInAny(head.op, gy, head.in, nil)
	if head.post != nil && head.post.Len() > 0 {
		if err := applyLeafDW(head, projectGap(gy, head.post.Shape...), lr, true); err != nil {
			return err
		}
	}
	if n == 1 {
		return nil
	}
	acc := accelOf(op)
	idx := 0
	if acc != nil {
		idx = acc.sparse % (n - 1)
		acc.sparse++
	}
	src := proxy
	if src == nil || src.Len() == 0 {
		src = gy
	}
	leaf := leaves[idx]
	if leaf.post == nil || leaf.post.Len() == 0 {
		return nil
	}
	gi := projectGap(src, leaf.post.Shape...)
	return applyLeafDW(leaf, gi, lr, true)
}

func trainTweenSplitLinearLeaves[T core.Numeric](op any, leaves []tweenLeaf[T], gy *core.Tensor[T], lr float64, cached bool) error {
	n := len(leaves)
	if n == 0 {
		return nil
	}
	acc := accelOf(op)
	if cached && acc != nil && len(acc.cache) == n && acc.cacheNorm > 1e-12 && acc.tick%acc.every() != 0 {
		live := gapNorm(gy)
		scale := live / acc.cacheNorm
		acc.tick++
		for i, leaf := range leaves {
			if leaf.post == nil || leaf.post.Len() == 0 || i >= len(acc.cache) {
				continue
			}
			buf := acc.cache[i]
			if len(buf) != leaf.post.Len() {
				continue
			}
			gi := core.NewTensor[T](leaf.post.Shape...)
			for j := range gi.Data {
				gi.Data[j] = core.FromFloat64[T](buf[j] * scale)
			}
			scaleGap(gi, 1.0/float64(n))
			if err := applyLeafDW(leaf, gi, lr, true); err != nil {
				return err
			}
		}
		return nil
	}
	byOp := make(map[any]tweenLeaf[T], n)
	leafIdx := make(map[any]int, n)
	for i, leaf := range leaves {
		byOp[leaf.op] = leaf
		leafIdx[leaf.op] = i
	}
	var fill [][]float64
	if cached && acc != nil {
		fill = make([][]float64, n)
	}
	if _, err := linearWalkTape(op, gy, byOp, leafIdx, n, lr, fill); err != nil {
		return err
	}
	if cached && acc != nil {
		acc.cache = fill
		acc.cacheNorm = gapNorm(gy)
		acc.tick++
	}
	return nil
}

func gapNorm[T core.Numeric](g *core.Tensor[T]) float64 {
	if g == nil || g.Len() == 0 {
		return 0
	}
	var s float64
	for _, v := range g.Data {
		f := core.AsFloat64(v)
		s += f * f
	}
	return math.Sqrt(s / float64(g.Len()))
}

func linearWalkTape[T core.Numeric](op any, gyDown *core.Tensor[T], byOp map[any]tweenLeaf[T], leafIdx map[any]int, n int, lr float64, fill [][]float64) (*core.Tensor[T], error) {
	if op == nil || gyDown == nil {
		return gyDown, nil
	}
	switch v := op.(type) {
	case *View:
		return gyDown, nil
	case *Stack:
		gy := gyDown
		for i := len(v.Children) - 1; i >= 0; i-- {
			gx, err := linearWalkTape(v.Children[i], gy, byOp, leafIdx, n, lr, fill)
			if err != nil {
				return nil, err
			}
			if gx != nil {
				gy = gx
			}
		}
		return gy, nil
	case *sequential.Layer:
		gy := gyDown
		for i := len(v.Children) - 1; i >= 0; i-- {
			gx, err := linearWalkTape(v.Children[i], gy, byOp, leafIdx, n, lr, fill)
			if err != nil {
				return nil, err
			}
			if gx != nil {
				gy = gx
			}
		}
		return gy, nil
	case *residual.Layer:
		gy := gyDown
		for i := len(v.Children) - 1; i >= 0; i-- {
			gx, err := linearWalkTape(v.Children[i], gy, byOp, leafIdx, n, lr, fill)
			if err != nil {
				return nil, err
			}
			if gx != nil {
				gy = gx
			}
		}
		return addTensors(gy, gyDown), nil
	case *Layer:
		posts := make([]*core.Tensor[T], len(v.Branches))
		for i, br := range v.Branches {
			if leaf, ok := byOp[br]; ok && leaf.post != nil {
				posts[i] = leaf.post
			} else {
				posts[i] = gyDown
			}
		}
		bgy, err := splitCombineGrad(v, gyDown, posts)
		if err != nil {
			return nil, err
		}
		var acc *core.Tensor[T]
		for i, br := range v.Branches {
			gx, err := linearWalkTape(br, bgy[i], byOp, leafIdx, n, lr, fill)
			if err != nil {
				return nil, err
			}
			acc = addTensors(acc, gx)
		}
		return acc, nil
	default:
		leaf, ok := byOp[op]
		if !ok {
			return linearGradInAny(op, gyDown, nil, nil), nil
		}
		if leaf.post != nil && leaf.post.Len() > 0 {
			gi := projectGap(gyDown, leaf.post.Shape...)
			if fill != nil {
				if i, ok := leafIdx[op]; ok && i < len(fill) {
					buf := make([]float64, gi.Len())
					for j := range gi.Data {
						buf[j] = core.AsFloat64(gi.Data[j])
					}
					fill[i] = buf
				}
			}
			scaleGap(gi, 1.0/float64(n))
			if err := applyLeafDW(leaf, gi, lr, true); err != nil {
				return nil, err
			}
		}
		return linearGradInAny(leaf.op, gyDown, leaf.in, leaf.pre), nil
	}
}

func addTensors[T core.Numeric](acc, gx *core.Tensor[T]) *core.Tensor[T] {
	if gx == nil || gx.Len() == 0 {
		return acc
	}
	if acc == nil {
		out := core.NewTensor[T](gx.Shape...)
		copy(out.Data, gx.Data)
		return out
	}
	if acc.Len() != gx.Len() {
		return acc
	}
	for i := range acc.Data {
		acc.Data[i] = core.FromFloat64[T](core.AsFloat64(acc.Data[i]) + core.AsFloat64(gx.Data[i]))
	}
	return acc
}
