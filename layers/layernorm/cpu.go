package layernorm

import (
	"fmt"
	"math"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/simd"
)

func gammaBetaF64(l *Layer) (g, b []float64, err error) {
	g, err = l.Gamma.WireF64()
	if err != nil {
		return nil, nil, fmt.Errorf("layernorm gamma: %w", err)
	}
	b, err = l.Beta.WireF64()
	if err != nil {
		return nil, nil, fmt.Errorf("layernorm beta: %w", err)
	}
	if len(g) < l.Cfg.Dim || len(b) < l.Cfg.Dim {
		return nil, nil, fmt.Errorf("layernorm affine short")
	}
	return g[:l.Cfg.Dim], b[:l.Cfg.Dim], nil
}

// ForwardCPUTiled — per-token LayerNorm.
func ForwardCPUTiled[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	return forwardHost(l, input, false)
}

// BackwardCPUTiled — dγ,dβ and dx; pre is x̂.
func BackwardCPUTiled[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	return backwardHost(l, gradOut, input, pre, false)
}

func forwardHost[T core.Numeric](l *Layer, input *core.Tensor[T], useSIMD bool) (pre, post *core.Tensor[T], err error) {
	lay, err := parseLayout(l.Cfg.Dim, input)
	if err != nil {
		return nil, nil, err
	}
	g, b, err := gammaBetaF64(l)
	if err != nil {
		return nil, nil, err
	}
	dim := l.Cfg.Dim
	eps := l.Cfg.Eps
	nTok := tokens(lay)
	xHat := make([]T, nTok*dim)
	y := make([]T, nTok*dim)
	ones := make([]float32, dim)
	for i := range ones {
		ones[i] = 1
	}

	simdPath := useSIMD && simd.Enabled()
	var gF, bF, xf, xhF, yF []float32
	if simdPath {
		gF = make([]float32, dim)
		bF = make([]float32, dim)
		for i := 0; i < dim; i++ {
			gF[i] = float32(g[i])
			bF[i] = float32(b[i])
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
			sum := simd.DotTile(xf, ones, 0, dim, 0)
			sumSq := simd.DotTile(xf, xf, 0, dim, 0)
			mean := sum / float64(dim)
			var_ := sumSq/float64(dim) - mean*mean
			inv := float32(1.0 / math.Sqrt(var_+eps))
			simd.LayerNormScaleF32(xf, gF, bF, xhF, yF, float32(mean), inv, dim)
			for i := 0; i < dim; i++ {
				xHat[base+i] = core.FromFloat64[T](float64(xhF[i]))
				y[base+i] = core.FromFloat64[T](float64(yF[i]))
			}
			continue
		}
		var sum, sumSq float64
		for i := 0; i < dim; i++ {
			v := core.AsFloat64(row[i])
			sum += v
			sumSq += v * v
		}
		mean := sum / float64(dim)
		var_ := sumSq/float64(dim) - mean*mean
		inv := 1.0 / math.Sqrt(var_+eps)
		for i := 0; i < dim; i++ {
			xh := (core.AsFloat64(row[i]) - mean) * inv
			xHat[base+i] = core.FromFloat64[T](xh)
			y[base+i] = core.FromFloat64[T](xh*g[i] + b[i])
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
	g, _, err := gammaBetaF64(l)
	if err != nil {
		return nil, nil, err
	}
	dim := l.Cfg.Dim
	eps := l.Cfg.Eps
	nTok := tokens(lay)
	if pre == nil || pre.Len() < nTok*dim {
		return nil, nil, fmt.Errorf("layernorm: pre (x̂) missing")
	}

	dGamma := make([]float64, dim)
	dBeta := make([]float64, dim)
	dx := make([]T, nTok*dim)
	ones := make([]float32, dim)
	for i := range ones {
		ones[i] = 1
	}
	simdPath := useSIMD && simd.Enabled()
	var xf, scratch, outf []float32
	if simdPath {
		xf = make([]float32, dim)
		scratch = make([]float32, dim)
		outf = make([]float32, dim)
	}

	for t := 0; t < nTok; t++ {
		base := t * dim
		row := input.Data[base : base+dim]
		dy := gradOut.Data[base : base+dim]
		xHatRow := pre.Data[base : base+dim]

		var sum, sumSq float64
		if simdPath {
			for i := 0; i < dim; i++ {
				xf[i] = float32(core.AsFloat64(row[i]))
			}
			sum = simd.DotTile(xf, ones, 0, dim, 0)
			sumSq = simd.DotTile(xf, xf, 0, dim, 0)
		} else {
			for i := 0; i < dim; i++ {
				v := core.AsFloat64(row[i])
				sum += v
				sumSq += v * v
			}
		}
		mean := sum / float64(dim)
		var_ := sumSq/float64(dim) - mean*mean
		inv := 1.0 / math.Sqrt(var_+eps)

		dxh := make([]float64, dim)
		var sumDxh, sumDxhXh float64
		for i := 0; i < dim; i++ {
			xh := core.AsFloat64(xHatRow[i])
			d := core.AsFloat64(dy[i])
			dGamma[i] += d * xh
			dBeta[i] += d
			dxh[i] = g[i] * d
			sumDxh += dxh[i]
			sumDxhXh += dxh[i] * xh
		}
		n := float64(dim)
		if simdPath {
			for i := 0; i < dim; i++ {
				xh := core.AsFloat64(xHatRow[i])
				scratch[i] = float32(n*dxh[i] - sumDxh - xh*sumDxhXh)
			}
			simd.ScaleXHatF32(scratch, outf, float32(inv/n), dim)
			for i := 0; i < dim; i++ {
				dx[base+i] = core.FromFloat64[T](float64(outf[i]))
			}
		} else {
			for i := 0; i < dim; i++ {
				xh := core.AsFloat64(xHatRow[i])
				dx[base+i] = core.FromFloat64[T](inv / n * (n*dxh[i] - sumDxh - xh*sumDxhXh))
			}
		}
	}

	gradIn = reshapeOut(dx, lay)
	gradW = core.NewTensor[T](1, 2*dim)
	for i := 0; i < dim; i++ {
		gradW.Data[i] = core.FromFloat64[T](dGamma[i])
		gradW.Data[dim+i] = core.FromFloat64[T](dBeta[i])
	}
	return gradIn, gradW, nil
}
