package parallel

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
)

// ForwardCPUTiled runs branches then combines.
func ForwardCPUTiled[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	return forwardHost(l, input)
}

// BackwardCPUTiled recomputes branch tapes and distributes grads.
func BackwardCPUTiled[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	return backwardHost(l, gradOut, input, pre)
}

type layInfo struct {
	batch, rows, dim int
	shape            []int
}

func parseIn[T core.Numeric](cfg Config, input *core.Tensor[T]) (layInfo, error) {
	if input == nil || len(input.Data) == 0 {
		return layInfo{}, fmt.Errorf("parallel: empty input")
	}
	dim := cfg.Dim
	var batch, rows int
	var shape []int
	switch {
	case cfg.SeqLen > 0:
		if len(input.Shape) != 3 || input.Shape[1] != cfg.SeqLen || input.Shape[2] != dim {
			return layInfo{}, fmt.Errorf("parallel: need [batch,%d,%d], got %v", cfg.SeqLen, dim, input.Shape)
		}
		batch, rows = input.Shape[0], cfg.SeqLen
		shape = append([]int(nil), input.Shape...)
	case len(input.Shape) == 2 && input.Shape[1] == dim:
		batch, rows = input.Shape[0], 1
		shape = append([]int(nil), input.Shape...)
	case len(input.Shape) == 3 && input.Shape[2] == dim:
		batch, rows = input.Shape[0], input.Shape[1]
		shape = append([]int(nil), input.Shape...)
	default:
		// Polymorphic ops (CNN/RNN/…) may use non-[…,Dim] layouts; combine uses
		// batch=Shape[0] and feat=Len/batch. Dense branches still require a
		// standard layout above (or they hard-error in dense.Forward).
		if len(input.Shape) < 1 || input.Shape[0] <= 0 {
			return layInfo{}, fmt.Errorf("parallel: unsupported shape %v for dim %d", input.Shape, dim)
		}
		batch, rows = input.Shape[0], 1
		shape = append([]int(nil), input.Shape...)
	}
	if batch <= 0 || rows <= 0 {
		return layInfo{}, fmt.Errorf("parallel: invalid batch/rows")
	}
	return layInfo{batch: batch, rows: rows, dim: dim, shape: shape}, nil
}

func flatten[T core.Numeric](in *core.Tensor[T], lay layInfo) (*core.Tensor[T], error) {
	need := lay.batch * lay.rows * lay.dim
	if in == nil || in.Len() < need {
		return nil, fmt.Errorf("parallel: flatten needs %d elems (batch=%d rows=%d dim=%d), have %d",
			need, lay.batch, lay.rows, lay.dim, in.Len())
	}
	out := core.NewTensor[T](lay.batch*lay.rows, lay.dim)
	copy(out.Data, in.Data[:need])
	return out, nil
}

func needsFlatInput(l *Layer) bool {
	if l == nil {
		return false
	}
	if l.Gate != nil {
		return true
	}
	for _, ch := range l.Branches {
		if needsDenseFlat(ch) {
			return true
		}
	}
	return false
}

func unflattenFeat[T core.Numeric](flat *core.Tensor[T], lay layInfo, feat int) *core.Tensor[T] {
	var out *core.Tensor[T]
	if lay.rows == 1 && len(lay.shape) == 2 {
		out = core.NewTensor[T](lay.batch, feat)
	} else {
		out = core.NewTensor[T](lay.batch, lay.rows, feat)
	}
	copy(out.Data, flat.Data)
	return out
}

func branchFeat[T core.Numeric](o *core.Tensor[T], n int) (int, error) {
	if o == nil || n <= 0 {
		return 0, fmt.Errorf("parallel: empty branch output")
	}
	if o.Len()%n != 0 {
		return 0, fmt.Errorf("parallel: branch out len %d not divisible by n=%d", o.Len(), n)
	}
	return o.Len() / n, nil
}

func needsDenseFlat(op any) bool {
	_, ok := op.(*dense.Layer)
	return ok
}

func forwardHost[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if len(l.Branches) == 0 {
		out := core.NewTensor[T](input.Shape...)
		copy(out.Data, input.Data)
		return out, out, nil
	}
	lay, err := parseIn(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	var flat *core.Tensor[T]
	if needsFlatInput(l) {
		flat, err = flatten(input, lay)
		if err != nil {
			return nil, nil, err
		}
	}
	n := lay.batch * lay.rows
	nb := len(l.Branches)
	branchOut := make([]*core.Tensor[T], nb)
	feats := make([]int, nb)
	for i, ch := range l.Branches {
		var denseIn *core.Tensor[T]
		if needsDenseFlat(ch) {
			denseIn = flat
		}
		_, o, err := branchForward(ch, input, denseIn)
		if err != nil {
			return nil, nil, fmt.Errorf("parallel fwd branch %d: %w", i, err)
		}
		feat, err := branchFeat(o, n)
		if err != nil {
			return nil, nil, fmt.Errorf("parallel fwd branch %d: %w", i, err)
		}
		branchOut[i] = o
		feats[i] = feat
	}

	outDim, err := combineOutDim(l.Cfg.Combine, feats)
	if err != nil {
		return nil, nil, err
	}
	if l.Cfg.OutFeat > 0 {
		expected := l.Cfg.OutDim()
		if expected > 0 && expected != outDim {
			return nil, nil, fmt.Errorf("parallel: measured out %d != cfg.OutDim %d", outDim, expected)
		}
	} else {
		l.Cfg.OutFeat = feats[0]
		if l.Cfg.Combine == CombineConcat {
			// OutFeat stays per-branch width for equal-width branches; OutDim uses Branches*OutFeat.
			// For unequal concat widths, store measured total on Core only.
			equal := true
			for _, f := range feats[1:] {
				if f != feats[0] {
					equal = false
					break
				}
			}
			if !equal {
				l.Cfg.OutFeat = 0
			}
		}
		l.Core.OutputHeight = outDim
	}

	combined := core.NewTensor[T](n, outDim)
	switch l.Cfg.Combine {
	case CombineAdd:
		for _, o := range branchOut {
			for j := range combined.Data {
				combined.Data[j] += o.Data[j]
			}
		}
	case CombineAvg:
		inv := core.FromFloat64[T](1.0 / float64(nb))
		for _, o := range branchOut {
			for j := range combined.Data {
				combined.Data[j] += o.Data[j]
			}
		}
		for j := range combined.Data {
			combined.Data[j] *= inv
		}
	case CombineFilter:
		if l.Gate == nil {
			return nil, nil, fmt.Errorf("parallel: filter mode requires Gate")
		}
		_, gateOut, err := dense.Forward(l.Gate, flat)
		if err != nil {
			return nil, nil, fmt.Errorf("parallel gate fwd: %w", err)
		}
		feat := feats[0]
		for r := 0; r < n; r++ {
			logits := make([]float64, nb)
			for i := 0; i < nb; i++ {
				logits[i] = core.AsFloat64(gateOut.Data[r*nb+i])
			}
			w := softmaxF64(logits)
			base := r * feat
			for i, o := range branchOut {
				wi := w[i]
				for j := 0; j < feat; j++ {
					combined.Data[base+j] += core.FromFloat64[T](core.AsFloat64(o.Data[base+j]) * wi)
				}
			}
		}
	default: // concat
		for r := 0; r < n; r++ {
			off := r * outDim
			for i, o := range branchOut {
				f := feats[i]
				src := r * f
				copy(combined.Data[off:off+f], o.Data[src:src+f])
				off += f
			}
		}
	}
	pre = core.NewTensor[T](input.Shape...)
	copy(pre.Data, input.Data)
	post = unflattenFeat(combined, lay, outDim)
	return pre, post, nil
}

func combineOutDim(mode CombineMode, feats []int) (int, error) {
	if len(feats) == 0 {
		return 0, fmt.Errorf("parallel: no branch feats")
	}
	switch mode {
	case CombineConcat:
		sum := 0
		for _, f := range feats {
			sum += f
		}
		return sum, nil
	default:
		for i := 1; i < len(feats); i++ {
			if feats[i] != feats[0] {
				return 0, fmt.Errorf("parallel: %s requires equal branch widths, got %v", mode, feats)
			}
		}
		return feats[0], nil
	}
}

func backwardHost[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	_ = pre
	if len(l.Branches) == 0 {
		gi := core.NewTensor[T](input.Shape...)
		if gradOut != nil {
			copy(gi.Data, gradOut.Data)
		}
		return gi, nil, nil
	}
	lay, err := parseIn(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	var flat *core.Tensor[T]
	if needsFlatInput(l) {
		flat, err = flatten(input, lay)
		if err != nil {
			return nil, nil, err
		}
	}
	n := lay.batch * lay.rows
	nb := len(l.Branches)

	branchPres := make([]*core.Tensor[T], nb)
	branchOuts := make([]*core.Tensor[T], nb)
	feats := make([]int, nb)
	for i, ch := range l.Branches {
		var denseIn *core.Tensor[T]
		if needsDenseFlat(ch) {
			denseIn = flat
		}
		p, o, err := branchForward(ch, input, denseIn)
		if err != nil {
			return nil, nil, fmt.Errorf("parallel recompute branch %d: %w", i, err)
		}
		feat, err := branchFeat(o, n)
		if err != nil {
			return nil, nil, fmt.Errorf("parallel recompute branch %d: %w", i, err)
		}
		branchPres[i], branchOuts[i], feats[i] = p, o, feat
	}
	outDim, err := combineOutDim(l.Cfg.Combine, feats)
	if err != nil {
		return nil, nil, err
	}

	gyFlat := flattenFeat(gradOut, lay, outDim)
	gInAcc := make([]float64, n*lay.dim)
	dWs := make([]*core.Tensor[T], 0, nb+1)

	accumIn := func(gx *core.Tensor[T]) {
		if gx == nil {
			return
		}
		lim := len(gInAcc)
		if gx.Len() < lim {
			lim = gx.Len()
		}
		for j := 0; j < lim; j++ {
			gInAcc[j] += core.AsFloat64(gx.Data[j])
		}
	}

	bwdOne := func(i int, branchGy *core.Tensor[T]) error {
		ch := l.Branches[i]
		var denseIn *core.Tensor[T]
		if needsDenseFlat(ch) {
			denseIn = flat
		}
		gx, dw, err := branchBackward(ch, branchGy, input, denseIn, branchPres[i])
		if err != nil {
			return err
		}
		dWs = append(dWs, dw)
		accumIn(gx)
		return nil
	}

	switch l.Cfg.Combine {
	case CombineAdd, CombineAvg:
		scale := 1.0
		if l.Cfg.Combine == CombineAvg {
			scale = 1.0 / float64(nb)
		}
		feat := feats[0]
		scaled := core.NewTensor[T](n, feat)
		for j := range scaled.Data {
			scaled.Data[j] = core.FromFloat64[T](core.AsFloat64(gyFlat.Data[j]) * scale)
		}
		for i := 0; i < nb; i++ {
			if err := bwdOne(i, scaled); err != nil {
				return nil, nil, fmt.Errorf("parallel bwd branch %d: %w", i, err)
			}
		}
	case CombineFilter:
		if l.Gate == nil {
			return nil, nil, fmt.Errorf("parallel: filter mode requires Gate")
		}
		gatePre, gateOut, err := dense.Forward(l.Gate, flat)
		if err != nil {
			return nil, nil, err
		}
		feat := feats[0]
		gateLogitsGrad := core.NewTensor[T](n, nb)
		for r := 0; r < n; r++ {
			logits := make([]float64, nb)
			for i := 0; i < nb; i++ {
				logits[i] = core.AsFloat64(gateOut.Data[r*nb+i])
			}
			w := softmaxF64(logits)
			base := r * feat
			gy := make([]float64, nb)
			for i := 0; i < nb; i++ {
				var dot float64
				for j := 0; j < feat; j++ {
					dot += core.AsFloat64(gyFlat.Data[base+j]) * core.AsFloat64(branchOuts[i].Data[base+j])
				}
				gy[i] = dot
			}
			gLogits := softmaxBwd(gy, w)
			for i := 0; i < nb; i++ {
				gateLogitsGrad.Data[r*nb+i] = core.FromFloat64[T](gLogits[i])
			}
		}
		for i := 0; i < nb; i++ {
			scaled := core.NewTensor[T](n, feat)
			for r := 0; r < n; r++ {
				logits := make([]float64, nb)
				for k := 0; k < nb; k++ {
					logits[k] = core.AsFloat64(gateOut.Data[r*nb+k])
				}
				w := softmaxF64(logits)
				base := r * feat
				for j := 0; j < feat; j++ {
					scaled.Data[base+j] = core.FromFloat64[T](core.AsFloat64(gyFlat.Data[base+j]) * w[i])
				}
			}
			if err := bwdOne(i, scaled); err != nil {
				return nil, nil, fmt.Errorf("parallel filter bwd branch %d: %w", i, err)
			}
		}
		gxG, dwG, err := dense.Backward(l.Gate, gateLogitsGrad, flat, gatePre)
		if err != nil {
			return nil, nil, fmt.Errorf("parallel gate bwd: %w", err)
		}
		dWs = append(dWs, dwG)
		accumIn(gxG)
	default: // concat
		offset := 0
		for i := 0; i < nb; i++ {
			f := feats[i]
			branchGy := core.NewTensor[T](n, f)
			for r := 0; r < n; r++ {
				src := r*outDim + offset
				dst := r * f
				copy(branchGy.Data[dst:dst+f], gyFlat.Data[src:src+f])
			}
			offset += f
			if err := bwdOne(i, branchGy); err != nil {
				return nil, nil, fmt.Errorf("parallel concat bwd branch %d: %w", i, err)
			}
		}
	}

	gradIn = core.NewTensor[T](input.Shape...)
	for i := range gInAcc {
		if i < gradIn.Len() {
			gradIn.Data[i] = core.FromFloat64[T](gInAcc[i])
		}
	}
	total := 0
	for _, dw := range dWs {
		if dw != nil {
			total += dw.Len()
		}
	}
	gradW = core.NewTensor[T](total)
	off := 0
	for _, dw := range dWs {
		if dw == nil {
			continue
		}
		copy(gradW.Data[off:off+dw.Len()], dw.Data)
		off += dw.Len()
	}
	return gradIn, gradW, nil
}

func flattenFeat[T core.Numeric](in *core.Tensor[T], lay layInfo, feat int) *core.Tensor[T] {
	out := core.NewTensor[T](lay.batch*lay.rows, feat)
	n := lay.batch * lay.rows * feat
	if in == nil || in.Len() < n {
		return out
	}
	copy(out.Data, in.Data[:n])
	return out
}
