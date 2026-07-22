package qwenasr

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/openfluke/welvet/simd"
)

type decLayer struct {
	in, post, qn, kn    []float32
	q, k, v, o, g, u, d *Linear
	kc, vc              []float32
}

type decoder struct {
	c       DecoderConfig
	emb     *embedTable
	head    *Linear
	norm    []float32
	layers  []decLayer
	rope    *ropeCache
	useSIMD bool
	kvLen   int

	// reusable scratch (sized for the current forward call)
	h, xn, qAll, kAll, vAll, attn, mlpOut []float32
	gate, up                              []float32
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
	d.useSIMD = on && simd.Enabled()
	d.head.UseSIMD = d.useSIMD
	for i := range d.layers {
		l := &d.layers[i]
		for _, x := range []*Linear{l.q, l.k, l.v, l.o, l.g, l.u, l.d} {
			setLinearFuse(x, d.useSIMD)
		}
	}
}

func (d *decoder) reset() {
	d.kvLen = 0
	for i := range d.layers {
		d.layers[i].kc = d.layers[i].kc[:0]
		d.layers[i].vc = d.layers[i].vc[:0]
	}
}

func (d *decoder) ensureScratch(nNew int) {
	H := d.c.Hidden
	qDim := d.c.Heads * d.c.HeadDim
	kvDim := d.c.KVHeads * d.c.HeadDim
	need := func(buf *[]float32, n int) {
		if cap(*buf) < n {
			*buf = make([]float32, n)
		} else {
			*buf = (*buf)[:n]
		}
	}
	need(&d.h, nNew*H)
	need(&d.xn, nNew*H)
	need(&d.qAll, nNew*qDim)
	need(&d.kAll, nNew*kvDim)
	need(&d.vAll, nNew*kvDim)
	need(&d.attn, nNew*qDim)
	need(&d.mlpOut, nNew*H)
	need(&d.gate, nNew*d.c.Intermediate)
	need(&d.up, nNew*d.c.Intermediate)
}

// forward runs nNew token embeddings through the stack (batched prefill or
// single-token decode), appends KV, and returns the last-position hidden
// (pre-final-norm — call logits() for the LM head).
func (d *decoder) forward(embeds []float32, nNew int) []float32 {
	if nNew <= 0 {
		return nil
	}
	cfg := d.c
	H := cfg.Hidden
	hd := cfg.HeadDim
	nh := cfg.Heads
	nkv := cfg.KVHeads
	qDim := nh * hd
	kvDim := nkv * hd
	rep := nh / nkv
	eps := cfg.RMSEps
	scale := float32(1 / math.Sqrt(float64(hd)))
	base := d.kvLen
	useSIMD := d.useSIMD

	d.ensureScratch(nNew)
	h := d.h
	copy(h, embeds[:nNew*H])
	xn, qAll, kAll, vAll, attn, mlpOut := d.xn, d.qAll, d.kAll, d.vAll, d.attn, d.mlpOut
	gate, up := d.gate, d.up

	for li := range d.layers {
		l := &d.layers[li]
		for i := 0; i < nNew; i++ {
			copy(xn[i*H:(i+1)*H], h[i*H:(i+1)*H])
			rmsNorm(xn[i*H:(i+1)*H], l.in, H, eps)
		}
		l.q.forwardSeq(xn, qAll, nNew)
		l.k.forwardSeq(xn, kAll, nNew)
		l.v.forwardSeq(xn, vAll, nNew)
		for i := 0; i < nNew; i++ {
			pos := base + i
			q := qAll[i*qDim : (i+1)*qDim]
			k := kAll[i*kvDim : (i+1)*kvDim]
			for hh := 0; hh < nh; hh++ {
				rmsNorm(q[hh*hd:(hh+1)*hd], l.qn, hd, eps)
				d.rope.apply(q[hh*hd:(hh+1)*hd], pos)
			}
			for hh := 0; hh < nkv; hh++ {
				rmsNorm(k[hh*hd:(hh+1)*hd], l.kn, hd, eps)
				d.rope.apply(k[hh*hd:(hh+1)*hd], pos)
			}
		}
		l.kc = append(l.kc, kAll[:nNew*kvDim]...)
		l.vc = append(l.vc, vAll[:nNew*kvDim]...)
		kc, vc := l.kc, l.vc

		for i := range attn {
			attn[i] = 0
		}

		attnOne := func(i int) {
			qpos := base + i
			n := qpos + 1
			sc := make([]float32, n)
			for hh := 0; hh < nh; hh++ {
				kvh := hh / rep
				qh := qAll[i*qDim+hh*hd : i*qDim+hh*hd+hd]
				for tpos := 0; tpos < n; tpos++ {
					kh := kc[tpos*kvDim+kvh*hd : tpos*kvDim+kvh*hd+hd]
					if useSIMD {
						sc[tpos] = simd.DotF32(qh, kh) * scale
					} else {
						sc[tpos] = dot(qh, kh, hd) * scale
					}
				}
				softmax(sc)
				out := attn[i*qDim+hh*hd : i*qDim+hh*hd+hd]
				for tpos := 0; tpos < n; tpos++ {
					vh := vc[tpos*kvDim+kvh*hd : tpos*kvDim+kvh*hd+hd]
					if useSIMD {
						simd.SaxpyF32(out, sc[tpos], vh, hd)
					} else {
						w := sc[tpos]
						for z := 0; z < hd; z++ {
							out[z] += w * vh[z]
						}
					}
				}
			}
		}

		if nNew >= 8 && useSIMD {
			workers := runtime.GOMAXPROCS(0)
			if workers > nNew {
				workers = nNew
			}
			var wg sync.WaitGroup
			chunk := (nNew + workers - 1) / workers
			for w := 0; w < workers; w++ {
				lo := w * chunk
				hi := lo + chunk
				if hi > nNew {
					hi = nNew
				}
				if lo >= hi {
					break
				}
				wg.Add(1)
				go func(a, b int) {
					defer wg.Done()
					for i := a; i < b; i++ {
						attnOne(i)
					}
				}(lo, hi)
			}
			wg.Wait()
		} else {
			for i := 0; i < nNew; i++ {
				attnOne(i)
			}
		}

		l.o.forwardSeq(attn, mlpOut, nNew)
		for j := 0; j < nNew*H; j++ {
			h[j] += mlpOut[j]
		}
		for i := 0; i < nNew; i++ {
			copy(xn[i*H:(i+1)*H], h[i*H:(i+1)*H])
			rmsNorm(xn[i*H:(i+1)*H], l.post, H, eps)
		}
		l.g.forwardSeq(xn, gate, nNew)
		l.u.forwardSeq(xn, up, nNew)
		for j := range gate {
			gate[j] = silu(gate[j]) * up[j]
		}
		l.d.forwardSeq(gate, mlpOut, nNew)
		for j := 0; j < nNew*H; j++ {
			h[j] += mlpOut[j]
		}
	}
	d.kvLen = base + nNew

	last := make([]float32, H)
	copy(last, h[(nNew-1)*H:nNew*H])
	return last
}

func (d *decoder) step(h []float32, _ int) []float32 {
	return d.forward(h, 1)
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

func (d *decoder) embedInto(id int, dst []float32) {
	d.emb.row(id, dst)
}
