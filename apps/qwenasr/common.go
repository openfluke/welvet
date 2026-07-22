package qwenasr

import (
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/simd"
	"math"
)

type Linear struct {
	Out, In int
	W, B    []float32
	UseSIMD bool
}

func (l *Linear) forward(x, y []float32) {
	if l.UseSIMD && simd.Enabled() {
		dense.GemvF32SIMD(l.W, x, y, l.Out, l.In)
	} else {
		for o := 0; o < l.Out; o++ {
			y[o] = dot(l.W[o*l.In:(o+1)*l.In], x, l.In)
		}
	}
	for o := range y[:l.Out] {
		if l.B != nil {
			y[o] += l.B[o]
		}
	}
}
func (l *Linear) forwardSeq(x, y []float32, n int) {
	for i := 0; i < n; i++ {
		l.forward(x[i*l.In:], y[i*l.Out:])
	}
}
func setLinearFuse(l *Linear, on bool) {
	if l != nil {
		l.UseSIMD = on
	}
}
func rmsNorm(x, w []float32, d int, eps float64) {
	var s float64
	for _, v := range x[:d] {
		s += float64(v * v)
	}
	z := float32(1 / math.Sqrt(s/float64(d)+eps))
	for i := 0; i < d; i++ {
		x[i] *= z * w[i]
	}
}
func layerNorm(x, w, b []float32, d int, eps float64) {
	var m, v float64
	for _, a := range x[:d] {
		m += float64(a)
	}
	m /= float64(d)
	for _, a := range x[:d] {
		q := float64(a) - m
		v += q * q
	}
	z := 1 / math.Sqrt(v/float64(d)+eps)
	for i := 0; i < d; i++ {
		x[i] = float32((float64(x[i])-m)*z)*w[i] + b[i]
	}
}
func gelu(x float32) float32 { return float32(.5 * float64(x) * (1 + math.Erf(float64(x)/math.Sqrt2))) }
func silu(x float32) float32 { return x / (1 + float32(math.Exp(float64(-x)))) }
func dot(a, b []float32, n int) float32 {
	var s float32
	for i := 0; i < n; i++ {
		s += a[i] * b[i]
	}
	return s
}
func softmax(x []float32) {
	m := float32(math.Inf(-1))
	for _, v := range x {
		if v > m {
			m = v
		}
	}
	var s float32
	for i := range x {
		x[i] = float32(math.Exp(float64(x[i] - m)))
		s += x[i]
	}
	for i := range x {
		x[i] /= s
	}
}

type ropeCache struct {
	dim      int
	cos, sin [][]float32
}

func newRopeCache(d, n int, theta float64) *ropeCache {
	r := &ropeCache{d, make([][]float32, n), make([][]float32, n)}
	for p := 0; p < n; p++ {
		r.cos[p] = make([]float32, d/2)
		r.sin[p] = make([]float32, d/2)
		for i := 0; i < d/2; i++ {
			a := float64(p) / math.Pow(theta, float64(2*i)/float64(d))
			r.cos[p][i] = float32(math.Cos(a))
			r.sin[p][i] = float32(math.Sin(a))
		}
	}
	return r
}
func (r *ropeCache) apply(x []float32, pos int) {
	h := r.dim / 2
	for i := 0; i < h; i++ {
		a, b := x[i], x[i+h]
		x[i] = a*r.cos[pos][i] - b*r.sin[pos][i]
		x[i+h] = b*r.cos[pos][i] + a*r.sin[pos][i]
	}
}

type embedTable struct {
	Rows, Dim int
	W         []float32
}

func (e *embedTable) row(id int, dst []float32) { copy(dst, e.W[id*e.Dim:(id+1)*e.Dim]) }
