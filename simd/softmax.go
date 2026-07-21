package simd

import "math"

// SoftmaxF32 writes stable softmax(y) of x scaled by 1/temp into y for i in [0,n).
func SoftmaxF32(x, y []float32, n int, temp float32) {
	if n <= 0 || len(x) < n || len(y) < n {
		return
	}
	if temp == 0 {
		temp = 1
	}
	softmaxF32Go(x, y, n, temp)
}

func softmaxF32Go(x, y []float32, n int, temp float32) {
	invT := float64(temp)
	var maxLogit float64 = -1e38
	i := 0
	for ; i+4 <= n; i += 4 {
		for j := 0; j < 4; j++ {
			v := float64(x[i+j]) / invT
			if v > maxLogit {
				maxLogit = v
			}
		}
	}
	for ; i < n; i++ {
		v := float64(x[i]) / invT
		if v > maxLogit {
			maxLogit = v
		}
	}

	var sumExp float64
	i = 0
	for ; i+4 <= n; i += 4 {
		for j := 0; j < 4; j++ {
			e := math.Exp(float64(x[i+j])/invT - maxLogit)
			y[i+j] = float32(e)
			sumExp += e
		}
	}
	for ; i < n; i++ {
		e := math.Exp(float64(x[i])/invT - maxLogit)
		y[i] = float32(e)
		sumExp += e
	}
	if sumExp == 0 {
		return
	}
	invSum := float32(1.0 / sumExp)
	i = 0
	for ; i+4 <= n; i += 4 {
		y[i] *= invSum
		y[i+1] *= invSum
		y[i+2] *= invSum
		y[i+3] *= invSum
	}
	for ; i < n; i++ {
		y[i] *= invSum
	}
}

// SoftmaxBwdF32 writes gx[i] = (y[i]/temp) * (gy[i] - dot(gy,y)) for i in [0,n).
func SoftmaxBwdF32(gy, y, gx []float32, n int, temp float32) {
	if n <= 0 || len(gy) < n || len(y) < n || len(gx) < n {
		return
	}
	if temp == 0 {
		temp = 1
	}
	softmaxBwdF32Go(gy, y, gx, n, temp)
}

func softmaxBwdF32Go(gy, y, gx []float32, n int, temp float32) {
	var dotProd float64
	i := 0
	for ; i+4 <= n; i += 4 {
		dotProd += float64(gy[i])*float64(y[i]) +
			float64(gy[i+1])*float64(y[i+1]) +
			float64(gy[i+2])*float64(y[i+2]) +
			float64(gy[i+3])*float64(y[i+3])
	}
	for ; i < n; i++ {
		dotProd += float64(gy[i]) * float64(y[i])
	}
	scale := float64(1.0 / temp)
	i = 0
	for ; i+4 <= n; i += 4 {
		gx[i] = float32(scale * float64(y[i]) * (float64(gy[i]) - dotProd))
		gx[i+1] = float32(scale * float64(y[i+1]) * (float64(gy[i+1]) - dotProd))
		gx[i+2] = float32(scale * float64(y[i+2]) * (float64(gy[i+2]) - dotProd))
		gx[i+3] = float32(scale * float64(y[i+3]) * (float64(gy[i+3]) - dotProd))
	}
	for ; i < n; i++ {
		gx[i] = float32(scale * float64(y[i]) * (float64(gy[i]) - dotProd))
	}
}
