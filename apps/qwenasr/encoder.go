package qwenasr

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/openfluke/welvet/simd"
)

type encLayer struct {
	lnw, lnb, fnw, fnb []float32
	q, k, v, o, f1, f2 *Linear
}
type encoder struct {
	c            EncoderConfig
	c1, c2, c3   *Conv2d
	out          *Linear
	layers       []encLayer
	postw, postb []float32
	p1, p2       *Linear
	useSIMD      bool
}

func newEncoder(s *tensorStore, c EncoderConfig) (*encoder, error) {
	e := &encoder{c: c}
	var err error
	if e.c1, err = s.loadConv2d("thinker.audio_tower.conv2d1"); err != nil {
		return nil, err
	}
	if e.c2, err = s.loadConv2d("thinker.audio_tower.conv2d2"); err != nil {
		return nil, err
	}
	if e.c3, err = s.loadConv2d("thinker.audio_tower.conv2d3"); err != nil {
		return nil, err
	}
	if e.out, err = s.loadLinear("thinker.audio_tower.conv_out", false); err != nil {
		return nil, err
	}
	for i := 0; i < c.Layers; i++ {
		p := fmt.Sprintf("thinker.audio_tower.layers.%d", i)
		l := encLayer{}
		if l.lnw, err = s.loadVec(p + ".self_attn_layer_norm.weight"); err != nil {
			return nil, err
		}
		if l.lnb, err = s.loadVec(p + ".self_attn_layer_norm.bias"); err != nil {
			return nil, err
		}
		if l.fnw, err = s.loadVec(p + ".final_layer_norm.weight"); err != nil {
			return nil, err
		}
		if l.fnb, err = s.loadVec(p + ".final_layer_norm.bias"); err != nil {
			return nil, err
		}
		for _, z := range []struct {
			n string
			q **Linear
		}{{"self_attn.q_proj", &l.q}, {"self_attn.k_proj", &l.k}, {"self_attn.v_proj", &l.v}, {"self_attn.out_proj", &l.o}, {"fc1", &l.f1}, {"fc2", &l.f2}} {
			if *z.q, err = s.loadLinear(p+"."+z.n, true); err != nil {
				return nil, err
			}
		}
		e.layers = append(e.layers, l)
	}
	if e.postw, err = s.loadVec("thinker.audio_tower.ln_post.weight"); err != nil {
		return nil, err
	}
	if e.postb, err = s.loadVec("thinker.audio_tower.ln_post.bias"); err != nil {
		return nil, err
	}
	if e.p1, err = s.loadLinear("thinker.audio_tower.proj1", true); err != nil {
		return nil, err
	}
	e.p2, err = s.loadLinear("thinker.audio_tower.proj2", true)
	return e, err
}
func (e *encoder) SetSIMD(on bool) {
	e.useSIMD = on && simd.Enabled()
	setLinearFuse(e.out, e.useSIMD)
	setLinearFuse(e.p1, e.useSIMD)
	setLinearFuse(e.p2, e.useSIMD)
	for i := range e.layers {
		l := &e.layers[i]
		for _, x := range []*Linear{l.q, l.k, l.v, l.o, l.f1, l.f2} {
			setLinearFuse(x, e.useSIMD)
		}
	}
}
func conv(x []float32, h, w int, c *Conv2d) ([]float32, int, int) {
	oh, ow := (h+1)/2, (w+1)/2
	nPos := oh * ow
	kDim := c.In * 9
	// im2col (stride-2, pad-1, 3×3) then weight-stationary GEMM + GELU.
	col := make([]float32, nPos*kDim)
	im2colWorkers := runtime.GOMAXPROCS(0)
	if im2colWorkers > oh {
		im2colWorkers = oh
	}
	if im2colWorkers < 2 || oh < 8 {
		im2colRows(x, col, h, w, oh, ow, c.In, kDim, 0, oh)
	} else {
		var wg sync.WaitGroup
		chunk := (oh + im2colWorkers - 1) / im2colWorkers
		for wkr := 0; wkr < im2colWorkers; wkr++ {
			lo := wkr * chunk
			hi := lo + chunk
			if hi > oh {
				hi = oh
			}
			if lo >= hi {
				break
			}
			wg.Add(1)
			go func(a, b int) {
				defer wg.Done()
				im2colRows(x, col, h, w, oh, ow, c.In, kDim, a, b)
			}(lo, hi)
		}
		wg.Wait()
	}
	y := make([]float32, c.Out*nPos)
	workers := runtime.GOMAXPROCS(0)
	if workers > c.Out {
		workers = c.Out
	}
	if workers < 2 || c.Out < 32 {
		convGemmRows(c.W, c.B, col, y, 0, c.Out, kDim, nPos)
	} else {
		var wg sync.WaitGroup
		chunk := (c.Out + workers - 1) / workers
		for wkr := 0; wkr < workers; wkr++ {
			lo := wkr * chunk
			hi := lo + chunk
			if hi > c.Out {
				hi = c.Out
			}
			if lo >= hi {
				break
			}
			wg.Add(1)
			go func(a, b int) {
				defer wg.Done()
				convGemmRows(c.W, c.B, col, y, a, b, kDim, nPos)
			}(lo, hi)
		}
		wg.Wait()
	}
	return y, oh, ow
}

func im2colRows(x, col []float32, h, w, oh, ow, inCh, kDim, y0, y1 int) {
	for yy := y0; yy < y1; yy++ {
		for xx := 0; xx < ow; xx++ {
			base := (yy*ow + xx) * kDim
			for in := 0; in < inCh; in++ {
				off := base + in*9
				for ky := 0; ky < 3; ky++ {
					iy := yy*2 + ky - 1
					for kx := 0; kx < 3; kx++ {
						ix := xx*2 + kx - 1
						if iy >= 0 && iy < h && ix >= 0 && ix < w {
							col[off+ky*3+kx] = x[(in*h+iy)*w+ix]
						}
					}
				}
			}
		}
	}
}

func convGemmRows(w, b, col, y []float32, rowLo, rowHi, kDim, nPos int) {
	useSIMD := simd.Enabled()
	for o := rowLo; o < rowHi; o++ {
		wrow := w[o*kDim : (o+1)*kDim]
		bias := b[o]
		dst := y[o*nPos : (o+1)*nPos]
		for p := 0; p < nPos; p++ {
			var s float32
			if useSIMD {
				s = float32(simd.DotTile(col[p*kDim:(p+1)*kDim], wrow, 0, kDim, 0))
			} else {
				s = dot(col[p*kDim:(p+1)*kDim], wrow, kDim)
			}
			dst[p] = gelu(s + bias)
		}
	}
}
func (e *encoder) forward(mel []float32) []float32 {
	frames := len(mel) / melBins
	chunk := e.c.NWindow * 2
	var xs []float32
	var lens []int
	tConv := time.Duration(0)
	for st := 0; st < frames; st += chunk {
		n := chunk
		if st+n > frames {
			n = frames - st
		}
		in := make([]float32, melBins*n)
		for f := 0; f < melBins; f++ {
			copy(in[f*n:], mel[f*frames+st:f*frames+st+n])
		}
		t0 := time.Now()
		a, h, w := conv(in, melBins, n, e.c1)
		a, h, w = conv(a, h, w, e.c2)
		a, h, w = conv(a, h, w, e.c3)
		tConv += time.Since(t0)
		z := make([]float32, w*e.out.In)
		for t := 0; t < w; t++ {
			for ch := 0; ch < e.c.DownsampleHidden; ch++ {
				for f := 0; f < h; f++ {
					z[t*e.out.In+ch*h+f] = a[(ch*h+f)*w+t]
				}
			}
		}
		p := make([]float32, w*e.c.DModel)
		e.out.forwardSeq(z, p, w)
		for t := 0; t < w; t++ {
			for d := 0; d < e.c.DModel/2; d++ {
				v := float64(t) / math.Pow(10000, float64(d)/float64(e.c.DModel/2-1))
				p[t*e.c.DModel+d] += float32(math.Sin(v))
				p[t*e.c.DModel+e.c.DModel/2+d] += float32(math.Cos(v))
			}
		}
		xs = append(xs, p...)
		lens = append(lens, w)
	}
	n := len(xs) / e.c.DModel
	win := lens[0] * (e.c.NWindowInfer / chunk)
	t0 := time.Now()
	for _, l := range e.layers {
		norm := append([]float32(nil), xs...)
		for i := 0; i < n; i++ {
			layerNorm(norm[i*e.c.DModel:], l.lnw, l.lnb, e.c.DModel, 1e-5)
		}
		q, k, v := make([]float32, len(xs)), make([]float32, len(xs)), make([]float32, len(xs))
		l.q.forwardSeq(norm, q, n)
		l.k.forwardSeq(norm, k, n)
		l.v.forwardSeq(norm, v, n)
		a := make([]float32, len(xs))
		hd := e.c.DModel / e.c.Heads
		scale := float32(1 / math.Sqrt(float64(hd)))
		useSIMD := e.useSIMD
		for st := 0; st < n; st += win {
			en := st + win
			if en > n {
				en = n
			}
			wn := en - st
			attnWin := func(i int) {
				for head := 0; head < e.c.Heads; head++ {
					scores := make([]float32, wn)
					qh := q[i*e.c.DModel+head*hd : i*e.c.DModel+head*hd+hd]
					for j := st; j < en; j++ {
						kh := k[j*e.c.DModel+head*hd : j*e.c.DModel+head*hd+hd]
						scores[j-st] = dotN(qh, kh, hd, useSIMD) * scale
					}
					softmax(scores)
					out := a[i*e.c.DModel+head*hd : i*e.c.DModel+head*hd+hd]
					for j := st; j < en; j++ {
						vh := v[j*e.c.DModel+head*hd : j*e.c.DModel+head*hd+hd]
						if useSIMD {
							simd.SaxpyF32(out, scores[j-st], vh, hd)
						} else {
							w := scores[j-st]
							for d := 0; d < hd; d++ {
								out[d] += w * vh[d]
							}
						}
					}
				}
			}
			if wn >= 8 && useSIMD {
				workers := runtime.GOMAXPROCS(0)
				if workers > wn {
					workers = wn
				}
				var wg sync.WaitGroup
				chunk := (wn + workers - 1) / workers
				for wkr := 0; wkr < workers; wkr++ {
					lo := st + wkr*chunk
					hi := lo + chunk
					if hi > en {
						hi = en
					}
					if lo >= hi {
						break
					}
					wg.Add(1)
					go func(a0, b0 int) {
						defer wg.Done()
						for i := a0; i < b0; i++ {
							attnWin(i)
						}
					}(lo, hi)
				}
				wg.Wait()
			} else {
				for i := st; i < en; i++ {
					attnWin(i)
				}
			}
		}
		tmp := make([]float32, len(xs))
		l.o.forwardSeq(a, tmp, n)
		for i := range xs {
			xs[i] += tmp[i]
		}
		norm = append([]float32(nil), xs...)
		for i := 0; i < n; i++ {
			layerNorm(norm[i*e.c.DModel:], l.fnw, l.fnb, e.c.DModel, 1e-5)
		}
		ff := make([]float32, n*l.f1.Out)
		l.f1.forwardSeq(norm, ff, n)
		for i := range ff {
			ff[i] = gelu(ff[i])
		}
		tmp = make([]float32, len(xs))
		l.f2.forwardSeq(ff, tmp, n)
		for i := range xs {
			xs[i] += tmp[i]
		}
	}
	tLayers := time.Since(t0)
	for i := 0; i < n; i++ {
		layerNorm(xs[i*e.c.DModel:], e.postw, e.postb, e.c.DModel, 1e-5)
	}
	z := make([]float32, n*e.p1.Out)
	e.p1.forwardSeq(xs, z, n)
	for i := range z {
		z[i] = gelu(z[i])
	}
	out := make([]float32, n*e.p2.Out)
	e.p2.forwardSeq(z, out, n)
	fmt.Fprintf(os.Stderr, "  qwenasr enc: conv=%v layers=%v n=%d win=%d simd=%v\n",
		tConv.Round(time.Millisecond), tLayers.Round(time.Millisecond), n, win, e.useSIMD)
	return out
}
