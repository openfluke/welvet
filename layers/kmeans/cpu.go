package kmeans

import (
	"fmt"

	"github.com/openfluke/welvet/core"
)

func ForwardCPUTiled[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	return forwardHost(l, input)
}

func BackwardCPUTiled[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	return backwardHost(l, gradOut, input, pre)
}

type layInfo struct {
	batch, d, k int
	shape       []int
}

func parseIn[T core.Numeric](cfg Config, input *core.Tensor[T]) (layInfo, error) {
	if input == nil || len(input.Data) == 0 {
		return layInfo{}, fmt.Errorf("kmeans: empty input")
	}
	d := cfg.FeatureDim
	if len(input.Shape) < 1 || input.Shape[len(input.Shape)-1] != d {
		return layInfo{}, fmt.Errorf("kmeans: last dim want %d, shape %v", d, input.Shape)
	}
	batch := 1
	for i := 0; i < len(input.Shape)-1; i++ {
		batch *= input.Shape[i]
	}
	if batch <= 0 {
		return layInfo{}, fmt.Errorf("kmeans: invalid batch")
	}
	want := batch * d
	if len(input.Data) < want {
		return layInfo{}, fmt.Errorf("kmeans: data short")
	}
	return layInfo{batch: batch, d: d, k: cfg.NumClusters, shape: append([]int(nil), input.Shape...)}, nil
}

func forwardHost[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	lay, err := parseIn(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	w, err := l.Centers.Weights.FlattenF32()
	if err != nil {
		return nil, nil, fmt.Errorf("kmeans weights: %w", err)
	}
	if len(w) < lay.k*lay.d {
		return nil, nil, fmt.Errorf("kmeans: weight short")
	}
	temp := l.Cfg.temp()
	tempSq := temp * temp
	outDim := l.Cfg.OutDim()
	outShape := append(append([]int(nil), lay.shape[:len(lay.shape)-1]...), outDim)
	pre = core.NewTensor[T](outShape...)
	post = core.NewTensor[T](outShape...)
	act := l.Cfg.Activation

	for b := 0; b < lay.batch; b++ {
		xOff := b * lay.d
		logits := make([]float64, lay.k)
		for k := 0; k < lay.k; k++ {
			var sq float64
			cOff := k * lay.d
			for d := 0; d < lay.d; d++ {
				diff := core.AsFloat64(input.Data[xOff+d]) - float64(w[cOff+d])
				sq += diff * diff
			}
			logits[k] = -sq / (2 * tempSq)
		}
		a := softMax(logits)
		oOff := b * outDim
		if l.Cfg.OutputMode == OutputFeatures {
			for d := 0; d < lay.d; d++ {
				var v float64
				for k := 0; k < lay.k; k++ {
					v += a[k] * float64(w[k*lay.d+d])
				}
				pre.Data[oOff+d] = core.FromFloat64[T](v)
				post.Data[oOff+d] = core.Activate(pre.Data[oOff+d], act)
			}
		} else {
			for k := 0; k < lay.k; k++ {
				pre.Data[oOff+k] = core.FromFloat64[T](a[k])
				post.Data[oOff+k] = core.Activate(pre.Data[oOff+k], act)
			}
		}
	}
	return pre, post, nil
}

func backwardHost[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	lay, err := parseIn(l.Cfg, input)
	if err != nil {
		return nil, nil, err
	}
	if gradOut == nil || pre == nil {
		return nil, nil, fmt.Errorf("kmeans: nil gradOut/pre")
	}
	w, err := l.Centers.Weights.FlattenF32()
	if err != nil {
		return nil, nil, err
	}
	temp := l.Cfg.temp()
	tempSq := temp * temp
	outDim := l.Cfg.OutDim()
	act := l.Cfg.Activation
	gInAcc := make([]float64, lay.batch*lay.d)
	gWAcc := make([]float64, lay.k*lay.d)

	for b := 0; b < lay.batch; b++ {
		xOff := b * lay.d
		oOff := b * outDim
		// recompute assignments
		logits := make([]float64, lay.k)
		for k := 0; k < lay.k; k++ {
			var sq float64
			cOff := k * lay.d
			for d := 0; d < lay.d; d++ {
				diff := core.AsFloat64(input.Data[xOff+d]) - float64(w[cOff+d])
				sq += diff * diff
			}
			logits[k] = -sq / (2 * tempSq)
		}
		a := softMax(logits)

		gy := make([]float64, outDim)
		for i := 0; i < outDim; i++ {
			gy[i] = core.AsFloat64(gradOut.Data[oOff+i]) * core.AsFloat64(core.ActivateDeriv(pre.Data[oOff+i], act))
		}

		var gradLogits []float64
		if l.Cfg.OutputMode == OutputFeatures {
			// dL/da_k = ⟨gy, c_k⟩; also dL/dc from reconstruction
			gyA := make([]float64, lay.k)
			for k := 0; k < lay.k; k++ {
				var dot float64
				for d := 0; d < lay.d; d++ {
					dot += gy[d] * float64(w[k*lay.d+d])
				}
				gyA[k] = dot
			}
			gradLogits = softMaxBwd(gyA, a)
			for k := 0; k < lay.k; k++ {
				for d := 0; d < lay.d; d++ {
					gWAcc[k*lay.d+d] += a[k] * gy[d]
				}
			}
		} else {
			gradLogits = softMaxBwd(gy, a)
		}

		for k := 0; k < lay.k; k++ {
			cOff := k * lay.d
			for d := 0; d < lay.d; d++ {
				diff := core.AsFloat64(input.Data[xOff+d]) - float64(w[cOff+d])
				gWAcc[cOff+d] += gradLogits[k] * diff / tempSq
				gInAcc[xOff+d] -= gradLogits[k] * diff / tempSq
			}
		}
	}

	gradIn = core.NewTensor[T](lay.shape...)
	for i := range gInAcc {
		gradIn.Data[i] = core.FromFloat64[T](gInAcc[i])
	}
	gradW = core.NewTensor[T](lay.k, lay.d)
	for i := range gWAcc {
		gradW.Data[i] = core.FromFloat64[T](gWAcc[i])
	}
	return gradIn, gradW, nil
}
