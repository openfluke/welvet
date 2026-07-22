// Package qwentts is a native Welvet (pure Go, no Python runtime) port of the
// Qwen3-TTS-12Hz CustomVoice model: text -> Talker (Qwen3 LLM) -> Code Predictor
// (MTP) -> Speech Decoder (Snake/ConvNeXt vocoder) -> 24 kHz PCM.
//
// The public API mirrors the mosstts app so Octo can drive it identically.
package qwentts

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/model/hf"
	"github.com/openfluke/welvet/simd"
	"github.com/openfluke/welvet/webgpu"
)

// Linear is a dense row-major weight [Out*In] with optional bias [Out].
type Linear struct {
	Out, In int
	W       []float32 // [Out][In] row-major
	B       []float32 // optional length Out
	UseSIMD bool
	UseGPU  bool
}

func (l *Linear) gpuKey() uintptr {
	if l == nil || len(l.W) == 0 {
		return 0
	}
	return webgpu.BlobKey(unsafe.Pointer(&l.W[0]))
}

// forward computes y = W·x (+b). len(x) >= In, len(y) >= Out.
func (l *Linear) forward(x, y []float32) {
	if l == nil {
		return
	}
	in, out := l.In, l.Out
	w := l.W

	// Sticky WebGPU GEMV for large mats (gpu_fuse).
	const gpuMinElems = 512 * 1024
	if l.UseGPU && webgpu.Available() && out*in >= gpuMinElems && len(w) >= out*in {
		if err := webgpu.DenseGEMVF32Resident(l.gpuKey(), w, x, y, 1, in, out); err == nil {
			if l.B != nil {
				for o := 0; o < out; o++ {
					y[o] += l.B[o]
				}
			}
			return
		}
	}

	// SIMD GEMV (simd_fuse) — also fallback when GPU misses.
	if (l.UseSIMD || l.UseGPU) && simd.Enabled() {
		dense.GemvF32SIMD(w, x, y, out, in)
		if l.B != nil {
			for o := 0; o < out; o++ {
				y[o] += l.B[o]
			}
		}
		return
	}

	for i := 0; i < out; i++ {
		row := w[i*in : i*in+in]
		var acc float32
		for j := 0; j < in; j++ {
			acc += row[j] * x[j]
		}
		if l.B != nil {
			acc += l.B[i]
		}
		y[i] = acc
	}
}

func setLinearFuse(l *Linear, simdOn, gpuOn bool) {
	if l != nil {
		l.UseSIMD, l.UseGPU = simdOn, gpuOn
	}
}

func setLinearFuseMany(simdOn, gpuOn bool, ls ...*Linear) {
	for _, l := range ls {
		setLinearFuse(l, simdOn, gpuOn)
	}
}

func warmLinearGPU(l *Linear) error {
	if l == nil || !l.UseGPU || !webgpu.Available() {
		return nil
	}
	const gpuMinElems = 512 * 1024
	if l.Out*l.In < gpuMinElems || len(l.W) < l.Out*l.In {
		return nil
	}
	return webgpu.WarmF32Weight(l.gpuKey(), l.W, l.Out, l.In)
}

// forwardSeq applies forward to each of seq rows in x [seq*In] -> y [seq*Out].
func (l *Linear) forwardSeq(x, y []float32, seq int) {
	for s := 0; s < seq; s++ {
		l.forward(x[s*l.In:(s+1)*l.In], y[s*l.Out:(s+1)*l.Out])
	}
}

// loadLinear loads name.weight (shape [out,in]) and optionally name.bias.
func loadLinear(path string, idx map[string]hf.TensorInfo, name string, bias bool) (*Linear, error) {
	ti, ok := idx[name+".weight"]
	if !ok {
		return nil, fmt.Errorf("missing %s.weight", name)
	}
	if len(ti.Shape) != 2 {
		return nil, fmt.Errorf("%s.weight: expected 2-D, got shape %v", name, ti.Shape)
	}
	w, err := hf.LoadF16Vector(path, idx, name+".weight")
	if err != nil {
		return nil, err
	}
	lin := &Linear{Out: ti.Shape[0], In: ti.Shape[1], W: w}
	if bias {
		b, err := hf.LoadF16Vector(path, idx, name+".bias")
		if err != nil {
			return nil, fmt.Errorf("%s.bias: %w", name, err)
		}
		lin.B = b
	}
	return lin, nil
}

// loadVec loads a 1-D tensor (norm weight, bias, alpha, etc.) as float32.
func loadVec(path string, idx map[string]hf.TensorInfo, name string) ([]float32, error) {
	return hf.LoadF16Vector(path, idx, name)
}

// embedTable is a lookup table [Rows][Dim] row-major.
type embedTable struct {
	Rows, Dim int
	W         []float32
}

func loadEmbed(path string, idx map[string]hf.TensorInfo, name string) (*embedTable, error) {
	ti, ok := idx[name]
	if !ok {
		return nil, fmt.Errorf("missing %s", name)
	}
	if len(ti.Shape) != 2 {
		return nil, fmt.Errorf("%s: expected 2-D embedding, got %v", name, ti.Shape)
	}
	w, err := hf.LoadF16Vector(path, idx, name)
	if err != nil {
		return nil, err
	}
	return &embedTable{Rows: ti.Shape[0], Dim: ti.Shape[1], W: w}, nil
}

// row returns embedding row id into dst (len >= Dim). Zero-fills on OOB.
func (e *embedTable) row(id int, dst []float32) {
	if id < 0 || id >= e.Rows {
		for i := 0; i < e.Dim; i++ {
			dst[i] = 0
		}
		return
	}
	copy(dst[:e.Dim], e.W[id*e.Dim:(id+1)*e.Dim])
}

// rmsNorm applies RMS normalization with optional affine weight over dim elems.
func rmsNorm(x, weight []float32, dim int, eps float64) {
	if dim <= 0 || len(x) < dim {
		return
	}
	var sumSq float64
	for i := 0; i < dim; i++ {
		v := float64(x[i])
		sumSq += v * v
	}
	inv := 1.0 / math.Sqrt(sumSq/float64(dim)+eps)
	for i := 0; i < dim; i++ {
		if weight != nil {
			x[i] = float32(float64(x[i]) * inv * float64(weight[i]))
		} else {
			x[i] = float32(float64(x[i]) * inv)
		}
	}
}

// rmsNormTo writes normed(x) into out without mutating x.
func rmsNormTo(out, x, weight []float32, dim int, eps float64) {
	copy(out[:dim], x[:dim])
	rmsNorm(out, weight, dim, eps)
}

// layerNorm applies LayerNorm with weight+bias over dim elems (in-place).
func layerNorm(x, weight, bias []float32, dim int, eps float64) {
	var mean float64
	for i := 0; i < dim; i++ {
		mean += float64(x[i])
	}
	mean /= float64(dim)
	var v float64
	for i := 0; i < dim; i++ {
		d := float64(x[i]) - mean
		v += d * d
	}
	inv := 1.0 / math.Sqrt(v/float64(dim)+eps)
	for i := 0; i < dim; i++ {
		nv := (float64(x[i]) - mean) * inv
		if weight != nil {
			nv *= float64(weight[i])
		}
		if bias != nil {
			nv += float64(bias[i])
		}
		x[i] = float32(nv)
	}
}

func silu(x float32) float32 {
	return x / float32(1+math.Exp(float64(-x)))
}

// siluMulInPlace applies silu in-place (no up multiply).
func siluMulInPlace(x []float32) {
	for i := range x {
		x[i] = silu(x[i])
	}
}

// siluMul writes silu(gate)*up into gate (in-place).
func siluMul(gate, up []float32) {
	n := len(gate)
	if n > len(up) {
		n = len(up)
	}
	if simd.Enabled() {
		simd.SiluMulF32(gate, up, gate, n)
		return
	}
	for j := 0; j < n; j++ {
		gate[j] = silu(gate[j]) * up[j]
	}
}

func gelu(x float32) float32 {
	// exact GELU (erf form) matching nn.GELU default
	return float32(0.5 * float64(x) * (1 + math.Erf(float64(x)/math.Sqrt2)))
}

func dotF32(a, b []float32, n int) float32 {
	if simd.Enabled() {
		return float32(simd.DotTile(a, b, 0, n, 0))
	}
	var acc float32
	for i := 0; i < n; i++ {
		acc += a[i] * b[i]
	}
	return acc
}

// softmaxInPlace normalizes scores[0:n] to a probability distribution.
func softmaxInPlace(scores []float32, n int) {
	if n <= 0 {
		return
	}
	if simd.Enabled() {
		simd.SoftmaxF32(scores[:n], scores[:n], n, 1)
		return
	}
	maxS := float32(math.Inf(-1))
	for i := 0; i < n; i++ {
		if scores[i] > maxS {
			maxS = scores[i]
		}
	}
	var sum float32
	for i := 0; i < n; i++ {
		e := float32(math.Exp(float64(scores[i] - maxS)))
		scores[i] = e
		sum += e
	}
	if sum == 0 {
		return
	}
	inv := 1 / sum
	for i := 0; i < n; i++ {
		scores[i] *= inv
	}
}

// ropeCache precomputes cos/sin for positions [0,maxPos) over headDim.
type ropeCache struct {
	headDim int
	cos     [][]float32 // [pos][headDim/2]
	sin     [][]float32
}

func newRopeCache(headDim, maxPos int, theta float64) *ropeCache {
	half := headDim / 2
	c := &ropeCache{headDim: headDim, cos: make([][]float32, maxPos), sin: make([][]float32, maxPos)}
	invFreq := make([]float64, half)
	for i := 0; i < half; i++ {
		invFreq[i] = 1.0 / math.Pow(theta, float64(2*i)/float64(headDim))
	}
	for p := 0; p < maxPos; p++ {
		c.cos[p] = make([]float32, half)
		c.sin[p] = make([]float32, half)
		for i := 0; i < half; i++ {
			ang := float64(p) * invFreq[i]
			c.cos[p][i] = float32(math.Cos(ang))
			c.sin[p][i] = float32(math.Sin(ang))
		}
	}
	return c
}

// applyRope rotates one head vector q[0:headDim] at position pos (HF rotate_half).
func (c *ropeCache) applyRope(q []float32, pos int) {
	half := c.headDim / 2
	cs := c.cos[pos]
	sn := c.sin[pos]
	for i := 0; i < half; i++ {
		a := q[i]
		b := q[i+half]
		q[i] = a*cs[i] - b*sn[i]
		q[i+half] = b*cs[i] + a*sn[i]
	}
}
