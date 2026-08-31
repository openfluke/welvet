package parallel

import (
	"fmt"
	"math"
	"sort"

	"github.com/openfluke/welvet/core"
)

// combineEqualWidth merges same-width branch posts (add/avg/max/sparsek/disagree).
// Used by CPU forward and Split-tape combineCollected.
func combineEqualWidth[T core.Numeric](l *Layer, branchOut []*core.Tensor[T]) (*core.Tensor[T], error) {
	if l == nil || len(branchOut) == 0 || branchOut[0] == nil {
		return nil, fmt.Errorf("parallel: combineEqualWidth empty")
	}
	nb := len(branchOut)
	out := core.NewTensor[T](branchOut[0].Shape...)
	n := out.Len()
	for _, o := range branchOut {
		if o == nil || o.Len() != n {
			return nil, fmt.Errorf("parallel: unequal branch widths for %s", l.Cfg.Combine)
		}
	}
	switch l.Cfg.Combine {
	case CombineAdd:
		for _, o := range branchOut {
			for j := range out.Data {
				out.Data[j] += o.Data[j]
			}
		}
	case CombineAvg:
		inv := core.FromFloat64[T](1.0 / float64(nb))
		for _, o := range branchOut {
			for j := range out.Data {
				out.Data[j] += o.Data[j]
			}
		}
		for j := range out.Data {
			out.Data[j] *= inv
		}
	case CombineMax:
		for j := 0; j < n; j++ {
			best := core.AsFloat64(branchOut[0].Data[j])
			for i := 1; i < nb; i++ {
				v := core.AsFloat64(branchOut[i].Data[j])
				if v > best {
					best = v
				}
			}
			out.Data[j] = core.FromFloat64[T](best)
		}
	case CombineSparseK:
		k := l.Cfg.sparseK()
		if k > nb {
			k = nb
		}
		type scored struct {
			i int
			n float64
		}
		scores := make([]scored, nb)
		for i, o := range branchOut {
			var s float64
			for j := 0; j < n; j++ {
				v := core.AsFloat64(o.Data[j])
				s += v * v
			}
			scores[i] = scored{i: i, n: math.Sqrt(s)}
		}
		sort.Slice(scores, func(a, b int) bool { return scores[a].n > scores[b].n })
		inv := core.FromFloat64[T](1.0 / float64(k))
		for t := 0; t < k; t++ {
			o := branchOut[scores[t].i]
			for j := range out.Data {
				out.Data[j] += o.Data[j]
			}
		}
		for j := range out.Data {
			out.Data[j] *= inv
		}
	case CombineDisagree:
		beta := l.Cfg.disagreeBeta()
		inv := 1.0 / float64(nb)
		mean := make([]float64, n)
		for _, o := range branchOut {
			for j := 0; j < n; j++ {
				mean[j] += core.AsFloat64(o.Data[j]) * inv
			}
		}
		if nb == 2 {
			for j := 0; j < n; j++ {
				a := core.AsFloat64(branchOut[0].Data[j])
				b := core.AsFloat64(branchOut[1].Data[j])
				out.Data[j] = core.FromFloat64[T](mean[j] + beta*(a-b))
			}
		} else {
			// avg + β·mean_i(self − mean) = avg (identity); use β·(cam0−mean) as accent.
			for j := 0; j < n; j++ {
				a0 := core.AsFloat64(branchOut[0].Data[j])
				out.Data[j] = core.FromFloat64[T](mean[j] + beta*(a0-mean[j]))
			}
		}
	default:
		return nil, fmt.Errorf("parallel: combineEqualWidth unsupported %s", l.Cfg.Combine)
	}
	return out, nil
}

// sparseKSelected returns top-K branch indices by ‖out‖₂ (descending).
func sparseKSelected[T core.Numeric](branchOut []*core.Tensor[T], k int) []int {
	nb := len(branchOut)
	if k > nb {
		k = nb
	}
	if k < 1 {
		k = 1
	}
	type scored struct {
		i int
		n float64
	}
	scores := make([]scored, nb)
	for i, o := range branchOut {
		var s float64
		if o != nil {
			for _, v := range o.Data {
				f := core.AsFloat64(v)
				s += f * f
			}
		}
		scores[i] = scored{i: i, n: math.Sqrt(s)}
	}
	sort.Slice(scores, func(a, b int) bool { return scores[a].n > scores[b].n })
	out := make([]int, k)
	for t := 0; t < k; t++ {
		out[t] = scores[t].i
	}
	return out
}
