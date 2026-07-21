package wav2vec2

import (
	"math"
	"runtime"
	"sync"
)

// Exact GELU (erf), matching torch.nn.functional.gelu default.
func gelu(x float32) float32 {
	xf := float64(x)
	return float32(0.5 * xf * (1 + math.Erf(xf/math.Sqrt2)))
}

func layerNorm(x []float32, w, b []float32, eps float64) {
	n := len(x)
	var mean float64
	for _, v := range x {
		mean += float64(v)
	}
	mean /= float64(n)
	var varSum float64
	for _, v := range x {
		d := float64(v) - mean
		varSum += d * d
	}
	inv := 1 / math.Sqrt(varSum/float64(n)+eps)
	for i := range x {
		y := (float64(x[i]) - mean) * inv
		if i < len(w) {
			y *= float64(w[i])
		}
		if i < len(b) {
			y += float64(b[i])
		}
		x[i] = float32(y)
	}
}

// groupNormPerChannel: GroupNorm(C groups, C channels) over time.
func groupNormPerChannel(x []float32, channels, time int, w, b []float32, eps float64) {
	for c := 0; c < channels; c++ {
		base := c * time
		var mean float64
		for t := 0; t < time; t++ {
			mean += float64(x[base+t])
		}
		mean /= float64(time)
		var varSum float64
		for t := 0; t < time; t++ {
			d := float64(x[base+t]) - mean
			varSum += d * d
		}
		inv := 1 / math.Sqrt(varSum/float64(time)+eps)
		scale := float64(1)
		shift := float64(0)
		if c < len(w) {
			scale = float64(w[c])
		}
		if c < len(b) {
			shift = float64(b[c])
		}
		for t := 0; t < time; t++ {
			x[base+t] = float32(((float64(x[base+t]) - mean) * inv) * scale + shift)
		}
	}
}

// conv1dValid: weight [out][in][k], input [in][Tin], output [out][Tout].
func conv1dValid(weight []float32, outC, inC, k, stride int, in []float32, inT int) ([]float32, int) {
	outT := (inT-k)/stride + 1
	if outT < 1 {
		return nil, 0
	}
	out := make([]float32, outC*outT)
	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = 1
	}
	if outC < workers {
		workers = outC
	}
	var wg sync.WaitGroup
	chunk := (outC + workers - 1) / workers
	for w := 0; w < workers; w++ {
		oc0 := w * chunk
		oc1 := oc0 + chunk
		if oc0 >= outC {
			break
		}
		if oc1 > outC {
			oc1 = outC
		}
		wg.Add(1)
		go func(oc0, oc1 int) {
			defer wg.Done()
			for oc := oc0; oc < oc1; oc++ {
				wBase := oc * inC * k
				oBase := oc * outT
				for t := 0; t < outT; t++ {
					inStart := t * stride
					var sum float32
					for ic := 0; ic < inC; ic++ {
						inBase := ic*inT + inStart
						wk := wBase + ic*k
						for j := 0; j < k; j++ {
							sum += weight[wk+j] * in[inBase+j]
						}
					}
					out[oBase+t] = sum
				}
			}
		}(oc0, oc1)
	}
	wg.Wait()
	return out, outT
}

func conv1dGrouped(weight, bias []float32, outC, inPerGroup, k, groups, pad int, in []float32, inT int, trimEven bool) []float32 {
	inC := groups * inPerGroup
	paddedT := inT + 2*pad
	padded := make([]float32, inC*paddedT)
	for c := 0; c < inC; c++ {
		copy(padded[c*paddedT+pad:], in[c*inT:(c+1)*inT])
	}
	outT := paddedT - k + 1
	out := make([]float32, outC*outT)
	outPerGroup := outC / groups
	for g := 0; g < groups; g++ {
		for oc := 0; oc < outPerGroup; oc++ {
			o := g*outPerGroup + oc
			wBase := o * inPerGroup * k
			for t := 0; t < outT; t++ {
				var sum float32
				if bias != nil {
					sum = bias[o]
				}
				for ic := 0; ic < inPerGroup; ic++ {
					inCh := g*inPerGroup + ic
					inBase := inCh * paddedT
					wk := wBase + ic*k
					for j := 0; j < k; j++ {
						sum += weight[wk+j] * padded[inBase+t+j]
					}
				}
				out[o*outT+t] = sum
			}
		}
	}
	if trimEven && k%2 == 0 && outT > 0 {
		newT := outT - 1
		trimmed := make([]float32, outC*newT)
		for o := 0; o < outC; o++ {
			copy(trimmed[o*newT:(o+1)*newT], out[o*outT:o*outT+newT])
		}
		return trimmed
	}
	return out
}

func linear(x, w, b []float32, t, in, out int) []float32 {
	y := make([]float32, t*out)
	workers := runtime.GOMAXPROCS(0)
	if workers > t {
		workers = t
	}
	if workers < 1 {
		workers = 1
	}
	var wg sync.WaitGroup
	chunk := (t + workers - 1) / workers
	for worker := 0; worker < workers; worker++ {
		i0 := worker * chunk
		i1 := i0 + chunk
		if i0 >= t {
			break
		}
		if i1 > t {
			i1 = t
		}
		wg.Add(1)
		go func(i0, i1 int) {
			defer wg.Done()
			for i := i0; i < i1; i++ {
				xb := i * in
				yb := i * out
				inRow := x[xb : xb+in]
				for o := 0; o < out; o++ {
					wb := o * in
					var sum float32
					if b != nil {
						sum = b[o]
					}
					row := w[wb : wb+in]
					for j := 0; j < in; j++ {
						sum += inRow[j] * row[j]
					}
					y[yb+o] = sum
				}
			}
		}(i0, i1)
	}
	wg.Wait()
	return y
}

func addInPlace(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}

func transposeCHWToTHC(x []float32, c, t int) []float32 {
	out := make([]float32, c*t)
	for ti := 0; ti < t; ti++ {
		for ci := 0; ci < c; ci++ {
			out[ti*c+ci] = x[ci*t+ti]
		}
	}
	return out
}

func transposeTHCToCHW(x []float32, t, c int) []float32 {
	out := make([]float32, c*t)
	for ti := 0; ti < t; ti++ {
		for ci := 0; ci < c; ci++ {
			out[ci*t+ti] = x[ti*c+ci]
		}
	}
	return out
}
