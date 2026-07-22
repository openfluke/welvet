package qwenasr

import (
	"fmt"
	"math"
)

type decLayer struct {
	in, post, qn, kn    []float32
	q, k, v, o, g, u, d *Linear
	kc, vc              []float32
}
type decoder struct {
	c      DecoderConfig
	emb    *embedTable
	head   *Linear
	norm   []float32
	layers []decLayer
	rope   *ropeCache
}

func newDecoder(s *tensorStore, c DecoderConfig) (*decoder, error) {
	w, sh, e := s.tensor("thinker.model.embed_tokens.weight")
	if e != nil {
		return nil, e
	}
	d := &decoder{c: c, emb: &embedTable{sh[0], sh[1], w}, rope: newRopeCache(c.HeadDim, 32768, c.RopeTheta)}
	d.norm, e = s.loadVec("thinker.model.norm.weight")
	if e != nil {
		return nil, e
	}
	if d.head, e = s.loadLinear("thinker.lm_head", false); e != nil {
		d.head = &Linear{c.Vocab, c.Hidden, w, nil, false}
	}
	for i := 0; i < c.Layers; i++ {
		p := fmt.Sprintf("thinker.model.layers.%d", i)
		l := decLayer{}
		for _, z := range []struct {
			n string
			p *[]float32
		}{{"input_layernorm.weight", &l.in}, {"post_attention_layernorm.weight", &l.post}, {"self_attn.q_norm.weight", &l.qn}, {"self_attn.k_norm.weight", &l.kn}} {
			if *z.p, e = s.loadVec(p + "." + z.n); e != nil {
				return nil, e
			}
		}
		for _, z := range []struct {
			n string
			p **Linear
		}{{"self_attn.q_proj", &l.q}, {"self_attn.k_proj", &l.k}, {"self_attn.v_proj", &l.v}, {"self_attn.o_proj", &l.o}, {"mlp.gate_proj", &l.g}, {"mlp.up_proj", &l.u}, {"mlp.down_proj", &l.d}} {
			if *z.p, e = s.loadLinear(p+"."+z.n, false); e != nil {
				return nil, e
			}
		}
		d.layers = append(d.layers, l)
	}
	return d, nil
}
func (d *decoder) SetSIMD(on bool) {
	d.head.UseSIMD = on
	for i := range d.layers {
		l := &d.layers[i]
		for _, x := range []*Linear{l.q, l.k, l.v, l.o, l.g, l.u, l.d} {
			setLinearFuse(x, on)
		}
	}
}
func (d *decoder) reset() {
	for i := range d.layers {
		d.layers[i].kc = nil
		d.layers[i].vc = nil
	}
}
func (d *decoder) step(h []float32, pos int) []float32 {
	H, hd := d.c.Hidden, d.c.HeadDim
	qDim := d.c.Heads * hd
	kvDim := d.c.KVHeads * hd
	for li := range d.layers {
		l := &d.layers[li]
		x := append([]float32(nil), h...)
		rmsNorm(x, l.in, H, d.c.RMSEps)
		q := make([]float32, qDim)
		k := make([]float32, kvDim)
		v := make([]float32, kvDim)
		l.q.forward(x, q)
		l.k.forward(x, k)
		l.v.forward(x, v)
		for head := 0; head < d.c.Heads; head++ {
			rmsNorm(q[head*hd:(head+1)*hd], l.qn, hd, d.c.RMSEps)
			d.rope.apply(q[head*hd:(head+1)*hd], pos)
		}
		for head := 0; head < d.c.KVHeads; head++ {
			rmsNorm(k[head*hd:(head+1)*hd], l.kn, hd, d.c.RMSEps)
			d.rope.apply(k[head*hd:(head+1)*hd], pos)
		}
		l.kc = append(l.kc, k...)
		l.vc = append(l.vc, v...)
		n := len(l.kc) / kvDim
		a := make([]float32, qDim)
		ratio := d.c.Heads / d.c.KVHeads
		scale := float32(1 / math.Sqrt(float64(hd)))
		for head := 0; head < d.c.Heads; head++ {
			kh := head / ratio
			sc := make([]float32, n)
			for j := 0; j < n; j++ {
				sc[j] = dot(q[head*hd:(head+1)*hd], l.kc[j*kvDim+kh*hd:], hd) * scale
			}
			softmax(sc)
			for j := 0; j < n; j++ {
				for z := 0; z < hd; z++ {
					a[head*hd+z] += sc[j] * l.vc[j*kvDim+kh*hd+z]
				}
			}
		}
		z := make([]float32, H)
		l.o.forward(a, z)
		for i := range h {
			h[i] += z[i]
		}
		x = append([]float32(nil), h...)
		rmsNorm(x, l.post, H, d.c.RMSEps)
		g, u := make([]float32, l.g.Out), make([]float32, l.u.Out)
		l.g.forward(x, g)
		l.u.forward(x, u)
		for i := range g {
			g[i] = silu(g[i]) * u[i]
		}
		l.d.forward(g, z)
		for i := range h {
			h[i] += z[i]
		}
	}
	return h
}
func (d *decoder) logits(h []float32) []float32 {
	x := append([]float32(nil), h...)
	rmsNorm(x, d.norm, d.c.Hidden, d.c.RMSEps)
	z := make([]float32, d.c.Vocab)
	d.head.forward(x, z)
	return z
}
func (d *decoder) embed(id int) []float32 {
	x := make([]float32, d.c.Hidden)
	d.emb.row(id, x)
	return x
}
