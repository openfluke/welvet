package rmsnorm

import (
	"fmt"
	"math"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/simd"
)

func gammaF64(l *Layer) ([]float64, error) {
	w, err := l.Gamma.WireF64()
	if err != nil {
		return nil, fmt.Errorf("rmsnorm gamma: %w", err)
	}
	if len(w) < l.Cfg.Dim {
		return nil, fmt.Errorf("rmsnorm gamma short")
	}
	return w[:l.Cfg.Dim], nil
}

// ForwardCPUTiled — per-token RMSNorm.
func ForwardCPUTiled[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	return forwardHost(l, input, false)
}

// BackwardCPUTiled — dγ and dx; pre is x̂.
func BackwardCPUTiled[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	return backwardHost(l, gradOut, input, pre, false)
}

func forwardHost[T core.Numeric](l *Layer, input *core.Tensor[T], useSIMD bool) (pre, post *core.Tensor[T], err error) {
	lay, err := parseLayout(l.Cfg.Dim, input)
	if err != nil {
		return nil, nil, err
	}
	g, err := gammaF64(l)
	if err != nil {
		return nil, nil, err
	}
	dim := l.Cfg.Dim
	eps := l.Cfg.Eps
	nTok := tokens(lay)
	xHat := make([]T, nTok*dim)
	y := make([]T, nTok*dim)

	simdPath := useSIMD && simd.Enabled()
	var gF, xf, xhF, yF []float32
	if simdPath {
		gF = make([]float32, dim)
		for i := 0; i < dim; i++ {
			gF[i] = float32(g[i])
		}
		xf = make([]float32, dim)
		xhF = make([]float32, dim)
		yF = make([]float32, dim)
	}

	for t := 0; t < nTok; t++ {
		base := t * dim
		row := input.Data[base : base+dim]
		if simdPath {
			for i := 0; i < dim; i++ {
				xf[i] = float32(core.AsFloat64(row[i]))
			}
			sumSq := simd.DotTile(xf, xf, 0, dim, 0)
			inv := float32(1.0 / math.Sqrt(sumSq/float64(dim)+eps))
			simd.RMSNormScaleF32(xf, gF, xhF, yF, inv, dim)
			for i := 0; i < dim; i++ {
				xHat[base+i] = core.FromFloat64[T](float64(xhF[i]))
				y[base+i] = core.FromFloat64[T](float64(yF[i]))
			}
			continue
		}
		var sumSq float64
		for i := 0; i < dim; i++ {
			v := core.AsFloat64(row[i])
			sumSq += v * v
		}
		inv := 1.0 / math.Sqrt(sumSq/float64(dim)+eps)
		for i := 0; i < dim; i++ {
			xh := core.AsFloat64(row[i]) * inv
			xHat[base+i] = core.FromFloat64[T](xh)
			y[base+i] = core.FromFloat64[T](xh * g[i])
		}
	}
	pre = reshapeOut(xHat, lay)
	post = reshapeOut(y, lay)
	return pre, post, nil
}

func backwardHost[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T], useSIMD bool) (gradIn, gradW *core.Tensor[T], err error) {
	lay, err := parseLayout(l.Cfg.Dim, input)
	if err != nil {
		return nil, nil, err
	}
	g, err := gammaF64(l)
	if err != nil {
		return nil, nil, err
	}
	dim := l.Cfg.Dim
	eps := l.Cfg.Eps
	nTok := tokens(lay)
	if pre == nil || pre.Len() < nTok*dim {
		return nil, nil, fmt.Errorf("rmsnorm: pre (x̂) missing")
	}

	dGamma := make([]float64, dim)
	dx := make([]T, nTok*dim)
	simdPath := useSIMD && simd.Enabled()
	var xf, uf, outf []float32
	if simdPath {
		xf = make([]float32, dim)
		uf = make([]float32, dim)
		outf = make([]float32, dim)
	}

	for t := 0; t < nTok; t++ {
		base := t * dim
		row := input.Data[base : base+dim]
		dy := gradOut.Data[base : base+dim]
		xHatRow := pre.Data[base : base+dim]

		var sumSq float64
		if simdPath {
			for i := 0; i < dim; i++ {
				xf[i] = float32(core.AsFloat64(row[i]))
			}
			sumSq = simd.DotTile(xf, xf, 0, dim, 0)
		} else {
			for i := 0; i < dim; i++ {
				v := core.AsFloat64(row[i])
				sumSq += v * v
			}
		}
		inv := 1.0 / math.Sqrt(sumSq/float64(dim)+eps)

		var mean float64
		u := make([]float64, dim)
		for i := 0; i < dim; i++ {
			xh := core.AsFloat64(xHatRow[i])
			d := core.AsFloat64(dy[i])
			dGamma[i] += d * xh
			u[i] = g[i] * d
			mean += xh * u[i]
		}
		mean /= float64(dim)
		if simdPath {
			for i := 0; i < dim; i++ {
				uf[i] = float32(u[i] - core.AsFloat64(xHatRow[i])*mean)
			}
			simd.ScaleXHatF32(uf, outf, float32(inv), dim)
			for i := 0; i < dim; i++ {
				dx[base+i] = core.FromFloat64[T](float64(outf[i]))
			}
		} else {
			for i := 0; i < dim; i++ {
				xh := core.AsFloat64(xHatRow[i])
				dx[base+i] = core.FromFloat64[T](inv * (u[i] - xh*mean))
			}
		}
	}

	gradIn = reshapeOut(dx, lay)
	gradW = core.NewTensor[T](1, dim)
	for i := 0; i < dim; i++ {
		gradW.Data[i] = core.FromFloat64[T](dGamma[i])
	}
	return gradIn, gradW, nil
}
