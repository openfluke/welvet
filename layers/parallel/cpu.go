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
	batch, rows, dim, outFeat, outDim int
	shape                             []int
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
		return layInfo{}, fmt.Errorf("parallel: unsupported shape %v for dim %d", input.Shape, dim)
	}
	if batch <= 0 || rows <= 0 {
		return layInfo{}, fmt.Errorf("parallel: invalid batch/rows")
	}
	return layInfo{batch: batch, rows: rows, dim: dim, outFeat: cfg.OutFeat, outDim: cfg.OutDim(), shape: shape}, nil
}

func flatten[T core.Numeric](in *core.Tensor[T], lay layInfo) *core.Tensor[T] {
	out := core.NewTensor[T](lay.batch*lay.rows, lay.dim)
	copy(out.Data, in.Data[:lay.batch*lay.rows*lay.dim])
	return out
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
	flat := flatten(input, lay)
	n := lay.batch * lay.rows
	branchOut := make([]*core.Tensor[T], len(l.Branches))
	for i, ch := range l.Branches {
		_, o, err := dense.Forward(ch, flat)
		if err != nil {
			return nil, nil, fmt.Errorf("parallel fwd branch %d: %w", i, err)
		}
		branchOut[i] = o
	}
	combined := core.NewTensor[T](n, lay.outDim)
	switch l.Cfg.Combine {
	case CombineAdd:
		for _, o := range branchOut {
			for j := range combined.Data {
				combined.Data[j] += o.Data[j]
			}
		}
	case CombineAvg:
		inv := core.FromFloat64[T](1.0 / float64(len(branchOut)))
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
		for r := 0; r < n; r++ {
			logits := make([]float64, len(l.Branches))
			for i := range l.Branches {
				logits[i] = core.AsFloat64(gateOut.Data[r*len(l.Branches)+i])
			}
			w := softmaxF64(logits)
			base := r * lay.outFeat
			for i, o := range branchOut {
				wi := w[i]
				for j := 0; j < lay.outFeat; j++ {
					combined.Data[base+j] += core.FromFloat64[T](core.AsFloat64(o.Data[base+j]) * wi)
				}
			}
		}
	default: // concat
		for r := 0; r < n; r++ {
			off := r * lay.outDim
			for i, o := range branchOut {
				src := r * lay.outFeat
				dst := off + i*lay.outFeat
				copy(combined.Data[dst:dst+lay.outFeat], o.Data[src:src+lay.outFeat])
			}
		}
	}
	// pre proxies input shape for tape; post is combined
	pre = core.NewTensor[T](input.Shape...)
	copy(pre.Data, input.Data)
	post = unflattenFeat(combined, lay, lay.outDim)
	return pre, post, nil
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
	flat := flatten(input, lay)
	n := lay.batch * lay.rows
	nb := len(l.Branches)

	branchIns := make([]*core.Tensor[T], nb)
	branchPres := make([]*core.Tensor[T], nb)
	branchOuts := make([]*core.Tensor[T], nb)
	for i, ch := range l.Branches {
		branchIns[i] = flat
		p, o, err := dense.Forward(ch, flat)
		if err != nil {
			return nil, nil, fmt.Errorf("parallel recompute branch %d: %w", i, err)
		}
		branchPres[i], branchOuts[i] = p, o
	}

	gyFlat := flattenFeat(gradOut, lay, lay.outDim)
	gInAcc := make([]float64, n*lay.dim)
	dWs := make([]*core.Tensor[T], 0, nb+1)

	switch l.Cfg.Combine {
	case CombineAdd, CombineAvg:
		scale := 1.0
		if l.Cfg.Combine == CombineAvg {
			scale = 1.0 / float64(nb)
		}
		scaled := core.NewTensor[T](n, lay.outFeat)
		for j := range scaled.Data {
			scaled.Data[j] = core.FromFloat64[T](core.AsFloat64(gyFlat.Data[j]) * scale)
		}
		for i := 0; i < nb; i++ {
			gx, dw, err := dense.Backward(l.Branches[i], scaled, branchIns[i], branchPres[i])
			if err != nil {
				return nil, nil, fmt.Errorf("parallel bwd branch %d: %w", i, err)
			}
			dWs = append(dWs, dw)
			for j := range gInAcc {
				gInAcc[j] += core.AsFloat64(gx.Data[j])
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
		gateLogitsGrad := core.NewTensor[T](n, nb)
		for r := 0; r < n; r++ {
			logits := make([]float64, nb)
			for i := 0; i < nb; i++ {
				logits[i] = core.AsFloat64(gateOut.Data[r*nb+i])
			}
			w := softmaxF64(logits)
			base := r * lay.outFeat
			gy := make([]float64, nb)
			// dL/dw_i = ⟨gy, branchOut_i⟩
			for i := 0; i < nb; i++ {
				var dot float64
				for j := 0; j < lay.outFeat; j++ {
					dot += core.AsFloat64(gyFlat.Data[base+j]) * core.AsFloat64(branchOuts[i].Data[base+j])
				}
				gy[i] = dot
			}
			gLogits := softmaxBwd(gy, w)
			for i := 0; i < nb; i++ {
				gateLogitsGrad.Data[r*nb+i] = core.FromFloat64[T](gLogits[i])
			}
			// branch grads: gy * w_i
			for i := 0; i < nb; i++ {
				scaled := core.NewTensor[T](1, lay.outFeat) // per-row path below
				_ = scaled
			}
		}
		// Per-branch backward with weighted gy
		for i := 0; i < nb; i++ {
			scaled := core.NewTensor[T](n, lay.outFeat)
			for r := 0; r < n; r++ {
				logits := make([]float64, nb)
				for k := 0; k < nb; k++ {
					logits[k] = core.AsFloat64(gateOut.Data[r*nb+k])
				}
				w := softmaxF64(logits)
				base := r * lay.outFeat
				for j := 0; j < lay.outFeat; j++ {
					scaled.Data[base+j] = core.FromFloat64[T](core.AsFloat64(gyFlat.Data[base+j]) * w[i])
				}
			}
			gx, dw, err := dense.Backward(l.Branches[i], scaled, branchIns[i], branchPres[i])
			if err != nil {
				return nil, nil, fmt.Errorf("parallel filter bwd branch %d: %w", i, err)
			}
			dWs = append(dWs, dw)
			for j := range gInAcc {
				gInAcc[j] += core.AsFloat64(gx.Data[j])
			}
		}
		gxG, dwG, err := dense.Backward(l.Gate, gateLogitsGrad, flat, gatePre)
		if err != nil {
			return nil, nil, fmt.Errorf("parallel gate bwd: %w", err)
		}
		dWs = append(dWs, dwG)
		for j := range gInAcc {
			gInAcc[j] += core.AsFloat64(gxG.Data[j])
		}
	default: // concat
		offset := 0
		for i := 0; i < nb; i++ {
			branchGy := core.NewTensor[T](n, lay.outFeat)
			for r := 0; r < n; r++ {
				src := r*lay.outDim + offset
				dst := r * lay.outFeat
				copy(branchGy.Data[dst:dst+lay.outFeat], gyFlat.Data[src:src+lay.outFeat])
			}
			offset += lay.outFeat
			gx, dw, err := dense.Backward(l.Branches[i], branchGy, branchIns[i], branchPres[i])
			if err != nil {
				return nil, nil, fmt.Errorf("parallel concat bwd branch %d: %w", i, err)
			}
			dWs = append(dWs, dw)
			for j := range gInAcc {
				gInAcc[j] += core.AsFloat64(gx.Data[j])
			}
		}
	}

	gradIn = core.NewTensor[T](input.Shape...)
	for i := range gInAcc {
		gradIn.Data[i] = core.FromFloat64[T](gInAcc[i])
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
