package simd

// ScaleMulGammaF32 writes out[i] = (x[i] * inv) * gamma[i] for i in [0,n).
// RMSNorm / LayerNorm forward affine after computing invRMS or invStd.
func ScaleMulGammaF32(x, gamma, out []float32, inv float32, n int) {
	if n <= 0 || len(x) < n || len(gamma) < n || len(out) < n {
		return
	}
	i := 0
	inv64 := float64(inv)
	for ; i+4 <= n; i += 4 {
		out[i] = float32(float64(x[i]) * inv64 * float64(gamma[i]))
		out[i+1] = float32(float64(x[i+1]) * inv64 * float64(gamma[i+1]))
		out[i+2] = float32(float64(x[i+2]) * inv64 * float64(gamma[i+2]))
		out[i+3] = float32(float64(x[i+3]) * inv64 * float64(gamma[i+3]))
	}
	for ; i < n; i++ {
		out[i] = float32(float64(x[i]) * inv64 * float64(gamma[i]))
	}
}

// ScaleXHatF32 writes xHat[i] = x[i] * inv (centered x already in x for LN).
func ScaleXHatF32(x, xHat []float32, inv float32, n int) {
	if n <= 0 || len(x) < n || len(xHat) < n {
		return
	}
	i := 0
	inv64 := float64(inv)
	for ; i+4 <= n; i += 4 {
		xHat[i] = float32(float64(x[i]) * inv64)
		xHat[i+1] = float32(float64(x[i+1]) * inv64)
		xHat[i+2] = float32(float64(x[i+2]) * inv64)
		xHat[i+3] = float32(float64(x[i+3]) * inv64)
	}
	for ; i < n; i++ {
		xHat[i] = float32(float64(x[i]) * inv64)
	}
}

// AffineGammaBetaF32 writes out[i] = xHat[i]*gamma[i] + beta[i].
func AffineGammaBetaF32(xHat, gamma, beta, out []float32, n int) {
	if n <= 0 || len(xHat) < n || len(gamma) < n || len(beta) < n || len(out) < n {
		return
	}
	i := 0
	for ; i+4 <= n; i += 4 {
		out[i] = float32(float64(xHat[i])*float64(gamma[i]) + float64(beta[i]))
		out[i+1] = float32(float64(xHat[i+1])*float64(gamma[i+1]) + float64(beta[i+1]))
		out[i+2] = float32(float64(xHat[i+2])*float64(gamma[i+2]) + float64(beta[i+2]))
		out[i+3] = float32(float64(xHat[i+3])*float64(gamma[i+3]) + float64(beta[i+3]))
	}
	for ; i < n; i++ {
		out[i] = float32(float64(xHat[i])*float64(gamma[i]) + float64(beta[i]))
	}
}

// SubScalarF32 writes out[i] = x[i] - mean.
func SubScalarF32(x, out []float32, mean float32, n int) {
	if n <= 0 || len(x) < n || len(out) < n {
		return
	}
	m := float64(mean)
	for i := 0; i < n; i++ {
		out[i] = float32(float64(x[i]) - m)
	}
}

// RMSNormScaleF32 writes xHat and y for one token: xHat=x*inv, y=xHat*gamma.
func RMSNormScaleF32(x, gamma, xHat, y []float32, inv float32, n int) {
	if n <= 0 {
		return
	}
	ScaleXHatF32(x, xHat, inv, n)
	ScaleMulGammaF32(x, gamma, y, inv, n)
}

// LayerNormScaleF32 writes xHat=(x-mean)*inv and y=xHat*gamma+beta for one token.
func LayerNormScaleF32(x, gamma, beta, xHat, y []float32, mean, inv float32, n int) {
	if n <= 0 || len(x) < n || len(gamma) < n || len(beta) < n || len(xHat) < n || len(y) < n {
		return
	}
	m, inv64 := float64(mean), float64(inv)
	i := 0
	for ; i+4 <= n; i += 4 {
		xh0 := (float64(x[i]) - m) * inv64
		xh1 := (float64(x[i+1]) - m) * inv64
		xh2 := (float64(x[i+2]) - m) * inv64
		xh3 := (float64(x[i+3]) - m) * inv64
		xHat[i], xHat[i+1], xHat[i+2], xHat[i+3] = float32(xh0), float32(xh1), float32(xh2), float32(xh3)
		y[i] = float32(xh0*float64(gamma[i]) + float64(beta[i]))
		y[i+1] = float32(xh1*float64(gamma[i+1]) + float64(beta[i+1]))
		y[i+2] = float32(xh2*float64(gamma[i+2]) + float64(beta[i+2]))
		y[i+3] = float32(xh3*float64(gamma[i+3]) + float64(beta[i+3]))
	}
	for ; i < n; i++ {
		xh := (float64(x[i]) - m) * inv64
		xHat[i] = float32(xh)
		y[i] = float32(xh*float64(gamma[i]) + float64(beta[i]))
	}
}
