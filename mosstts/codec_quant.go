package mosstts

import (
	"fmt"
	"math"
)

type conv1d struct {
	OutCh, InCh, K int
	W              []float32 // [out][in][k]
	B              []float32
}

func (c *conv1d) Forward(x []float32, length int) []float32 {
	// x: [inCh][length], y: [outCh][length]; kernel size 1 (or general k with same-pad left for causal — k=1 only here)
	out := make([]float32, c.OutCh*length)
	for t := 0; t < length; t++ {
		for o := 0; o < c.OutCh; o++ {
			var acc float32
			if c.B != nil {
				acc = c.B[o]
			}
			base := o * c.InCh * c.K
			for i := 0; i < c.InCh; i++ {
				for kk := 0; kk < c.K; kk++ {
					// centered? k=1 always for LFQ projs
					srcT := t + kk - (c.K - 1) // causal left-pad style; for k=1 → t
					if srcT < 0 || srcT >= length {
						continue
					}
					acc += c.W[base+i*c.K+kk] * x[i*length+srcT]
				}
			}
			out[o*length+t] = acc
		}
	}
	return out
}

func fuseWeightNorm(g, v []float32, outCh, inCh, k int) []float32 {
	w := make([]float32, outCh*inCh*k)
	for o := 0; o < outCh; o++ {
		var sum float64
		off := o * inCh * k
		for i := 0; i < inCh*k; i++ {
			vv := float64(v[off+i])
			sum += vv * vv
		}
		inv := 1.0 / math.Sqrt(sum+1e-12)
		gv := float64(g[o])
		if len(g) == outCh*1*1 {
			gv = float64(g[o])
		}
		for i := 0; i < inCh*k; i++ {
			w[off+i] = float32(gv * float64(v[off+i]) * inv)
		}
	}
	return w
}

func loadWNConv1d(tensors map[string][]float32, prefix string, outCh, inCh, k int) (*conv1d, error) {
	g, okG := tensors[prefix+".parametrizations.weight.original0"]
	v, okV := tensors[prefix+".parametrizations.weight.original1"]
	if !okG || !okV {
		// plain weight
		w, ok := tensors[prefix+".weight"]
		if !ok {
			return nil, fmt.Errorf("missing conv weight %s", prefix)
		}
		b := tensors[prefix+".bias"]
		return &conv1d{OutCh: outCh, InCh: inCh, K: k, W: w, B: b}, nil
	}
	w := fuseWeightNorm(g, v, outCh, inCh, k)
	b := tensors[prefix+".bias"]
	return &conv1d{OutCh: outCh, InCh: inCh, K: k, W: w, B: b}, nil
}

type lfqQuantizer struct {
	Codebook []float32 // [size][dim]
	Size     int
	Dim      int
	OutProj  *conv1d // codebook_dim → rvq_dim
}

func (q *lfqQuantizer) DecodeCode(ids []int) []float32 {
	// returns [rvq_dim][T]
	T := len(ids)
	// emb [cbDim][T]
	emb := make([]float32, q.Dim*T)
	for t, id := range ids {
		if id < 0 {
			id = 0
		}
		if id >= q.Size {
			id = q.Size - 1
		}
		src := q.Codebook[id*q.Dim : (id+1)*q.Dim]
		for d := 0; d < q.Dim; d++ {
			emb[d*T+t] = src[d]
		}
	}
	if q.OutProj == nil {
		return emb
	}
	return q.OutProj.Forward(emb, T)
}

type residualLFQ struct {
	Quantizers []*lfqQuantizer
	OutputProj *conv1d
	RVQDim     int
	OutputDim  int
}

func (r *residualLFQ) DecodeCodes(codes [][]int) []float32 {
	nq := len(codes)
	T := len(codes[0])
	emb := make([]float32, r.RVQDim*T)
	for i := 0; i < nq && i < len(r.Quantizers); i++ {
		zi := r.Quantizers[i].DecodeCode(codes[i])
		for j := range emb {
			emb[j] += zi[j]
		}
	}
	if r.OutputProj != nil {
		return r.OutputProj.Forward(emb, T)
	}
	return emb
}

func loadResidualLFQ(tensors map[string][]float32, prefix string, inputDim, rvqDim, outputDim, nQ, cbSize, cbDim int) (*residualLFQ, error) {
	_ = inputDim
	var outProj *conv1d
	var err error
	if rvqDim != outputDim {
		outProj, err = loadWNConv1d(tensors, prefix+".output_proj", outputDim, rvqDim, 1)
		if err != nil {
			return nil, err
		}
	}
	qs := make([]*lfqQuantizer, nQ)
	for i := 0; i < nQ; i++ {
		p := fmt.Sprintf("%s.quantizers.%d", prefix, i)
		cb, ok := tensors[p+".codebook.weight"]
		if !ok {
			return nil, fmt.Errorf("missing %s.codebook.weight", p)
		}
		var outP *conv1d
		if cbDim != rvqDim {
			outP, err = loadWNConv1d(tensors, p+".out_proj", rvqDim, cbDim, 1)
			if err != nil {
				return nil, err
			}
		}
		qs[i] = &lfqQuantizer{Codebook: cb, Size: cbSize, Dim: cbDim, OutProj: outP}
	}
	return &residualLFQ{Quantizers: qs, OutputProj: outProj, RVQDim: rvqDim, OutputDim: outputDim}, nil
}
