package simd

// SaxpyF32 computes y[i] += alpha * x[i] for i in [0,n).
// Used by vocoder causal/depthwise conv (shifted axpy over time).
func SaxpyF32(y []float32, alpha float32, x []float32, n int) {
	if n <= 0 || alpha == 0 || len(y) < n || len(x) < n {
		return
	}
	if simdEnabled() && n >= 8 {
		saxpyF32Simd(&y[0], alpha, &x[0], n)
		return
	}
	saxpyF32Go(y, alpha, x, n)
}

func saxpyF32Go(y []float32, alpha float32, x []float32, n int) {
	a := float64(alpha)
	i := 0
	for ; i+8 <= n; i += 8 {
		y[i+0] = float32(float64(y[i+0]) + a*float64(x[i+0]))
		y[i+1] = float32(float64(y[i+1]) + a*float64(x[i+1]))
		y[i+2] = float32(float64(y[i+2]) + a*float64(x[i+2]))
		y[i+3] = float32(float64(y[i+3]) + a*float64(x[i+3]))
		y[i+4] = float32(float64(y[i+4]) + a*float64(x[i+4]))
		y[i+5] = float32(float64(y[i+5]) + a*float64(x[i+5]))
		y[i+6] = float32(float64(y[i+6]) + a*float64(x[i+6]))
		y[i+7] = float32(float64(y[i+7]) + a*float64(x[i+7]))
	}
	for ; i < n; i++ {
		y[i] = float32(float64(y[i]) + a*float64(x[i]))
	}
}

// FillF32 writes y[i] = v for i in [0,n).
func FillF32(y []float32, v float32, n int) {
	if n <= 0 || len(y) < n {
		return
	}
	i := 0
	for ; i+8 <= n; i += 8 {
		y[i], y[i+1], y[i+2], y[i+3] = v, v, v, v
		y[i+4], y[i+5], y[i+6], y[i+7] = v, v, v, v
	}
	for ; i < n; i++ {
		y[i] = v
	}
}

// AddScaledF32Stride computes y[i*stride] += alpha * x[i] for i in [0,n).
// Used by transposed (upsampling) conv when writing every stride-th sample.
func AddScaledF32Stride(y []float32, yOff, stride int, alpha float32, x []float32, n int) {
	if n <= 0 || stride <= 0 || alpha == 0 || len(x) < n {
		return
	}
	if stride == 1 {
		if yOff+n > len(y) {
			return
		}
		SaxpyF32(y[yOff:yOff+n], alpha, x[:n], n)
		return
	}
	a := float64(alpha)
	end := yOff + (n-1)*stride
	if end >= len(y) || yOff < 0 {
		return
	}
	i := 0
	for ; i+4 <= n; i += 4 {
		p0 := yOff + i*stride
		p1 := p0 + stride
		p2 := p1 + stride
		p3 := p2 + stride
		y[p0] = float32(float64(y[p0]) + a*float64(x[i]))
		y[p1] = float32(float64(y[p1]) + a*float64(x[i+1]))
		y[p2] = float32(float64(y[p2]) + a*float64(x[i+2]))
		y[p3] = float32(float64(y[p3]) + a*float64(x[i+3]))
	}
	for ; i < n; i++ {
		p := yOff + i*stride
		y[p] = float32(float64(y[p]) + a*float64(x[i]))
	}
}
