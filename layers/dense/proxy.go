package dense

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/simd"
	"github.com/openfluke/welvet/weights"
)

// LinearGradIn is gx = Wᵀ gy with no ⊙ act'. MatVecT (same kernel as CPU
// backward's gx), never a W×W product. Any Numeric activation dtype.
func LinearGradIn[T core.Numeric](l *Layer, gy, input *core.Tensor[T]) (*core.Tensor[T], error) {
	if l == nil || l.Weights == nil || gy == nil || input == nil {
		return nil, fmt.Errorf("dense: LinearGradIn nil")
	}
	batch, in, out, err := dims(l, input)
	if err != nil {
		return nil, err
	}
	src := gy
	if gy.Len() != batch*out {
		src = reshapeGap(gy, batch, out)
	}
	gx := core.NewTensor[T](batch, in)
	for b := 0; b < batch; b++ {
		if err := weights.MatVecT(l.Weights, src.Data[b*out:(b+1)*out], gx.Data[b*in:(b+1)*in]); err != nil {
			return nil, err
		}
	}
	return gx, nil
}

// GradWOnly is dW = g xᵀ (optional ⊙ act'(pre)). No gx / Wᵀ. Hidden HeadProxy
// and FastProxy use this so they do not pay a second matvec they would discard.
func GradWOnly[T core.Numeric](l *Layer, gy, input, pre *core.Tensor[T]) (*core.Tensor[T], error) {
	if l == nil || l.Weights == nil || gy == nil || input == nil {
		return nil, fmt.Errorf("dense: GradWOnly nil")
	}
	batch, in, out, err := dims(l, input)
	if err != nil {
		return nil, err
	}
	src := gy
	if gy.Len() != batch*out {
		src = reshapeGap(gy, batch, out)
	}
	dPre := make([]float64, batch*out)
	act := l.Core.Activation
	useAct := pre != nil && pre.Len() >= batch*out
	for b := 0; b < batch; b++ {
		for o := 0; o < out; o++ {
			g := core.AsFloat64(src.Data[b*out+o])
			if useAct {
				g *= core.AsFloat64(core.ActivateDeriv(pre.Data[b*out+o], act))
			}
			dPre[b*out+o] = g
		}
	}
	dW := core.NewTensor[T](out, in)
	dW64 := make([]float64, out*in)
	for b := 0; b < batch; b++ {
		x32 := core.SliceAsFloat32(input.Data[b*in : (b+1)*in])
		for o := 0; o < out; o++ {
			g := dPre[b*out+o]
			if g == 0 {
				continue
			}
			simd.SaxpyF32AccF64(dW64[o*in:(o+1)*in], g, x32, in)
		}
	}
	core.SliceFromFloat64(dW64, dW.Data)
	return dW, nil
}

func reshapeGap[T core.Numeric](src *core.Tensor[T], batch, out int) *core.Tensor[T] {
	dst := core.NewTensor[T](batch, out)
	if src == nil || dst.Len() == 0 || src.Len() == 0 {
		return dst
	}
	sn, dn := src.Len(), dst.Len()
	if sn == dn {
		copy(dst.Data, src.Data)
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
