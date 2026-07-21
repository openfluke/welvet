package simd

import "math"

// SiluMulF32 computes out[i] = silu(gate[i]) * up[i] for i in [0,n).
// silu(x) = x / (1 + exp(-x)). Used by SwiGLU fused gate product.
func SiluMulF32(gate, up, out []float32, n int) {
	if n <= 0 || len(gate) < n || len(up) < n || len(out) < n {
		return
	}
	siluMulF32Go(gate, up, out, n)
}

func siluMulF32Go(gate, up, out []float32, n int) {
	i := 0
	for ; i+4 <= n; i += 4 {
		g0, g1, g2, g3 := float64(gate[i]), float64(gate[i+1]), float64(gate[i+2]), float64(gate[i+3])
		out[i] = float32(g0/(1+math.Exp(-g0)) * float64(up[i]))
		out[i+1] = float32(g1/(1+math.Exp(-g1)) * float64(up[i+1]))
		out[i+2] = float32(g2/(1+math.Exp(-g2)) * float64(up[i+2]))
		out[i+3] = float32(g3/(1+math.Exp(-g3)) * float64(up[i+3]))
	}
	for ; i < n; i++ {
		g := float64(gate[i])
		out[i] = float32(g / (1 + math.Exp(-g)) * float64(up[i]))
	}
}

// SiluMulBwdF32 writes dGate and dUp from dy given gate/up preactivations:
//
//	dUp = dy * silu(gate)
//	dGate = dy * up * dSilu(gate)
//
// where dSilu = σ(g) * (1 + g*(1-σ(g))).
func SiluMulBwdF32(gate, up, dy, dGate, dUp []float32, n int) {
	if n <= 0 || len(gate) < n || len(up) < n || len(dy) < n || len(dGate) < n || len(dUp) < n {
		return
	}
	siluMulBwdF32Go(gate, up, dy, dGate, dUp, n)
}

func siluMulBwdF32Go(gate, up, dy, dGate, dUp []float32, n int) {
	for i := 0; i < n; i++ {
		g := float64(gate[i])
		u := float64(up[i])
		d := float64(dy[i])
		sig := 1.0 / (1.0 + math.Exp(-g))
		silu := g * sig
		dSilu := sig * (1.0 + g*(1.0-sig))
		dUp[i] = float32(d * silu)
		dGate[i] = float32(d * u * dSilu)
	}
}
