package qwenasr

import (
	"fmt"
	"math"
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
	setLinearFuse(e.out, on)
	setLinearFuse(e.p1, on)
	setLinearFuse(e.p2, on)
	for i := range e.layers {
		l := &e.layers[i]
		for _, x := range []*Linear{l.q, l.k, l.v, l.o, l.f1, l.f2} {
			setLinearFuse(x, on)
		}
	}
}
func conv(x []float32, h, w int, c *Conv2d) ([]float32, int, int) {
	oh, ow := (h+1)/2, (w+1)/2
	y := make([]float32, c.Out*oh*ow)
	for o := 0; o < c.Out; o++ {
		for yy := 0; yy < oh; yy++ {
			for xx := 0; xx < ow; xx++ {
				v := c.B[o]
				for in := 0; in < c.In; in++ {
					for ky := 0; ky < 3; ky++ {
						for kx := 0; kx < 3; kx++ {
							iy, ix := yy*2+ky-1, xx*2+kx-1
							if iy >= 0 && iy < h && ix >= 0 && ix < w {
								v += c.W[((o*c.In+in)*3+ky)*3+kx] * x[(in*h+iy)*w+ix]
							}
						}
					}
				}
				y[(o*oh+yy)*ow+xx] = gelu(v)
			}
		}
	}
	return y, oh, ow
}
func (e *encoder) forward(mel []float32) []float32 {
	frames := len(mel) / melBins
	chunk := e.c.NWindow * 2
	var xs []float32
	var lens []int
	for st := 0; st < frames; st += chunk {
		n := chunk
		if st+n > frames {
			n = frames - st
		}
		in := make([]float32, melBins*n)
		for f := 0; f < melBins; f++ {
			copy(in[f*n:], mel[f*frames+st:f*frames+st+n])
		}
		a, h, w := conv(in, melBins, n, e.c1)
		a, h, w = conv(a, h, w, e.c2)
		a, h, w = conv(a, h, w, e.c3)
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
		for st := 0; st < n; st += win {
			en := st + win
			if en > n {
				en = n
			}
			for i := st; i < en; i++ {
				for head := 0; head < e.c.Heads; head++ {
					scores := make([]float32, en-st)
					for j := st; j < en; j++ {
						scores[j-st] = dot(q[i*e.c.DModel+head*hd:], k[j*e.c.DModel+head*hd:], hd) / float32(math.Sqrt(float64(hd)))
					}
					softmax(scores)
					for j := st; j < en; j++ {
						for d := 0; d < hd; d++ {
							a[i*e.c.DModel+head*hd+d] += scores[j-st] * v[j*e.c.DModel+head*hd+d]
						}
					}
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
	return out
}
