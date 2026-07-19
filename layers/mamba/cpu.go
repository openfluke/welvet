package mamba

import (
	"fmt"
	"math"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
)

func ForwardCPUTiled[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	return forwardHost(l, input)
}

func BackwardCPUTiled[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	return backwardHost(l, gradOut, input, pre)
}

type layInfo struct {
	batch, t, d, inner int
}

func parseIn[T core.Numeric](cfg Config, input *core.Tensor[T]) (layInfo, error) {
	if input == nil || len(input.Shape) != 3 {
		return layInfo{}, fmt.Errorf("mamba: need [batch,seq,dmodel], got %v", input.Shape)
	}
	if input.Shape[1] != cfg.SeqLen || input.Shape[2] != cfg.DModel {
		return layInfo{}, fmt.Errorf("mamba: shape %v != [%d,%d,%d]", input.Shape, input.Shape[0], cfg.SeqLen, cfg.DModel)
	}
	return layInfo{batch: input.Shape[0], t: cfg.SeqLen, d: cfg.DModel, inner: cfg.InnerDim()}, nil
}

func forwardHost[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	lay, err := parseIn(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	n := lay.batch * lay.t
	flat := core.NewTensor[T](n, lay.d)
	copy(flat.Data, input.Data)
	_, z, err := dense.Forward(l.InProj, flat)
	if err != nil {
		return nil, nil, fmt.Errorf("mamba in: %w", err)
	}
	// z = [x | dt] each Inner
	scanned := core.NewTensor[T](n, lay.inner)
	for b := 0; b < lay.batch; b++ {
		h := make([]float64, lay.inner)
		for t := 0; t < lay.t; t++ {
			row := (b*lay.t + t) * (2 * lay.inner)
			outRow := (b*lay.t + t) * lay.inner
			for i := 0; i < lay.inner; i++ {
				x := core.AsFloat64(z.Data[row+i])
				dt := softplus(core.AsFloat64(z.Data[row+lay.inner+i]))
				a := math.Exp(-dt * math.Exp(float64(l.ALog[i])))
				h[i] = a*h[i] + dt*x
				y := h[i] + float64(l.DSkip[i])*x
				scanned.Data[outRow+i] = core.FromFloat64[T](y)
			}
		}
	}
	preFlat, postFlat, err := dense.Forward(l.OutProj, scanned)
	if err != nil {
		return nil, nil, fmt.Errorf("mamba out: %w", err)
	}
	pre = core.NewTensor[T](lay.batch, lay.t, lay.d)
	post = core.NewTensor[T](lay.batch, lay.t, lay.d)
	copy(pre.Data, preFlat.Data)
	copy(post.Data, postFlat.Data)
	return pre, post, nil
}

func backwardHost[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	// Recompute forward intermediates; train OutProj + InProj via Dense; ALog/DSkip via finite path in dW tail.
	lay, err := parseIn(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	if gradOut == nil {
		return nil, nil, fmt.Errorf("mamba: nil gradOut")
	}
	_ = pre
	n := lay.batch * lay.t
	flat := core.NewTensor[T](n, lay.d)
	copy(flat.Data, input.Data)
	inPre, z, err := dense.Forward(l.InProj, flat)
	if err != nil {
		return nil, nil, err
	}
	scanned := core.NewTensor[T](n, lay.inner)
	hs := make([][]float64, lay.batch) // per-batch final not needed; store all h
	type step struct{ h, x, dt, a float64 }
	tape := make([][]step, lay.batch)
	for b := 0; b < lay.batch; b++ {
		h := make([]float64, lay.inner)
		tape[b] = make([]step, lay.t*lay.inner)
		for t := 0; t < lay.t; t++ {
			row := (b*lay.t + t) * (2 * lay.inner)
			outRow := (b*lay.t + t) * lay.inner
			for i := 0; i < lay.inner; i++ {
				x := core.AsFloat64(z.Data[row+i])
				dt := softplus(core.AsFloat64(z.Data[row+lay.inner+i]))
				a := math.Exp(-dt * math.Exp(float64(l.ALog[i])))
				hPrev := h[i]
				h[i] = a*h[i] + dt*x
				y := h[i] + float64(l.DSkip[i])*x
				scanned.Data[outRow+i] = core.FromFloat64[T](y)
				tape[b][t*lay.inner+i] = step{h: hPrev, x: x, dt: dt, a: a}
			}
		}
		hs[b] = h
	}
	outPre, _, err := dense.Forward(l.OutProj, scanned)
	if err != nil {
		return nil, nil, err
	}
	gyOut := core.NewTensor[T](n, lay.d)
	copy(gyOut.Data, gradOut.Data)
	gScan, dWOut, err := dense.Backward(l.OutProj, gyOut, scanned, outPre)
	if err != nil {
		return nil, nil, fmt.Errorf("mamba out bwd: %w", err)
	}

	gZ := core.NewTensor[T](n, 2*lay.inner)
	dA := make([]float64, lay.inner)
	dD := make([]float64, lay.inner)
	for b := 0; b < lay.batch; b++ {
		gh := make([]float64, lay.inner)
		for t := lay.t - 1; t >= 0; t-- {
			outRow := (b*lay.t + t) * lay.inner
			row := (b*lay.t + t) * (2 * lay.inner)
			for i := 0; i < lay.inner; i++ {
				st := tape[b][t*lay.inner+i]
				gy := core.AsFloat64(gScan.Data[outRow+i])
				// y = h_new + D*x; h_new = a*h_prev + dt*x
				dD[i] += gy * st.x
				gxY := gy * float64(l.DSkip[i])
				ghNew := gy + gh[i]
				ga := ghNew * st.h
				gdt := ghNew * st.x
				gx := ghNew*st.dt + gxY
				gh[i] = ghNew * st.a
				// a = exp(-dt * exp(ALog)); da/dALog = a * (-dt * exp(ALog))
				expA := math.Exp(float64(l.ALog[i]))
				dA[i] += ga * st.a * (-st.dt * expA)
				// dt = softplus(z_dt); d softplus = sigmoid
				sig := 1 / (1 + math.Exp(-core.AsFloat64(z.Data[row+lay.inner+i])))
				gZ.Data[row+i] = core.FromFloat64[T](gx)
				gZ.Data[row+lay.inner+i] = core.FromFloat64[T]((gdt + ga*st.a*(-expA)) * sig)
			}
		}
	}
	gxFlat, dWIn, err := dense.Backward(l.InProj, gZ, flat, inPre)
	if err != nil {
		return nil, nil, fmt.Errorf("mamba in bwd: %w", err)
	}
	gradIn = core.NewTensor[T](lay.batch, lay.t, lay.d)
	copy(gradIn.Data, gxFlat.Data)

	total := dWIn.Len() + dWOut.Len() + lay.inner*2
	gradW = core.NewTensor[T](total)
	off := 0
	copy(gradW.Data[off:], dWIn.Data)
	off += dWIn.Len()
	copy(gradW.Data[off:], dWOut.Data)
	off += dWOut.Len()
	for i := 0; i < lay.inner; i++ {
		gradW.Data[off+i] = core.FromFloat64[T](dA[i])
		gradW.Data[off+lay.inner+i] = core.FromFloat64[T](dD[i])
	}
	return gradIn, gradW, nil
}
