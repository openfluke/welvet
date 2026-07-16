package transformer

import (
	"fmt"
	"math"
	"time"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/dense"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/rmsnorm"
	"github.com/openfluke/welvet/swiglu"
	"github.com/openfluke/welvet/weights"
)

func (m *Model) isHybrid() bool {
	return m != nil && (m.Architecture == "qwen35_hybrid" || m.Architecture == "qwen3_dense" || m.embedPacked != nil)
}

// IsHybrid reports Qwen3.5 / Bonsai GDN+full-attn architecture.
func (m *Model) IsHybrid() bool { return m.isHybrid() }

func (m *Model) forwardTokensHybrid(ids []uint32) ([]float32, error) {
	if len(ids) == 0 {
		return nil, fmt.Errorf("transformer: empty ids")
	}
	hSize := m.HiddenSize
	useGPU := m.Exec.Backend == core.BackendWebGPU
	nTok := len(ids)
	t0 := time.Now()

	// Layer-major prefill: each resident weight is reused across all prompt tokens
	// before the next layer (far less thrash than token-outer on partial VRAM).
	hs := make([][]float32, nTok)
	for t := 0; t < nTok; t++ {
		hs[t] = make([]float32, hSize)
		if err := gatherEmbedPacked(m.embedPacked, ids[t:t+1], hs[t]); err != nil {
			return nil, err
		}
	}

	nBlocks := len(m.Blocks)
	for i := range m.Blocks {
		if !m.Quiet && nTok >= 4 && (i == 0 || (i+1)%4 == 0 || i+1 == nBlocks) {
			elapsed := time.Since(t0)
			rate := float64(i+1) / math.Max(elapsed.Seconds(), 1e-9)
			fmt.Printf("\r  loading prompt layers %d/%d (%.2f layer/s)\033[K", i+1, nBlocks, rate)
		}
		b := &m.Blocks[i]
		for t := 0; t < nTok; t++ {
			h := hs[t]
			n1t := core.NewTensor[float32](1, 1, hSize)
			copy(n1t.Data, h)
			_, n1, err := rmsnorm.Forward(b.AttnNorm, n1t)
			if err != nil {
				return nil, fmt.Errorf("block %d attn_norm: %w", i, err)
			}
			mix := make([]float32, hSize)
			switch b.LayerType {
			case "linear_attention":
				if b.GDN != nil {
					b.GDN.UseGPU = useGPU
				}
				if err := b.GDN.ForwardDecode(n1.Data, mix); err != nil {
					return nil, fmt.Errorf("block %d gdn: %w", i, err)
				}
			case "full_attention":
				if err := forwardFullAttnDecode(b, n1.Data, mix, m.MaxSeqLen); err != nil {
					return nil, fmt.Errorf("block %d attn: %w", i, err)
				}
			default:
				return nil, fmt.Errorf("block %d: unknown layer type %q", i, b.LayerType)
			}
			for j := range h {
				h[j] += mix[j]
			}
			n2t := core.NewTensor[float32](1, 1, hSize)
			copy(n2t.Data, h)
			_, n2, err := rmsnorm.Forward(b.FFNNorm, n2t)
			if err != nil {
				return nil, fmt.Errorf("block %d ffn_norm: %w", i, err)
			}
			_, f, err := swiglu.Forward(b.FFN, n2)
			if err != nil {
				return nil, fmt.Errorf("block %d ffn: %w", i, err)
			}
			for j := range h {
				h[j] += f.Data[j]
			}
		}
	}

	last := hs[nTok-1]
	if m.HasFinalNorm && m.FinalNorm != nil {
		ht := core.NewTensor[float32](1, 1, hSize)
		copy(ht.Data, last)
		_, normed, err := rmsnorm.Forward(m.FinalNorm, ht)
		if err != nil {
			return nil, err
		}
		copy(last, normed.Data)
	}
	if !m.Quiet && nTok >= 4 {
		fmt.Printf("\r\033[K")
	}
	return m.applyLMHead(last)
}

func forwardFullAttnDecode(b *Block, x, y []float32, maxSeq int) error {
	hd, nh, nkv := b.HeadDim, b.NumHeads, b.NumKVHeads
	qDim := nh * hd
	kvDim := nkv * hd
	qGateDim := qDim
	if b.OutputGate {
		qGateDim = qDim * 2
	}
	qOut := make([]float32, qGateDim)
	kOut := make([]float32, kvDim)
	vOut := make([]float32, kvDim)
	if err := matVecDense(b.Q, x, qOut); err != nil {
		return err
	}
	if err := matVecDense(b.K, x, kOut); err != nil {
		return err
	}
	if err := matVecDense(b.V, x, vOut); err != nil {
		return err
	}

	var gate []float32
	q := qOut
	if b.OutputGate {
		// q_proj rows are [heads, 2*head_dim] — per-head [q | gate], not [all q | all gate].
		q = make([]float32, qDim)
		gate = make([]float32, qDim)
		for h := 0; h < nh; h++ {
			base := h * 2 * hd
			copy(q[h*hd:(h+1)*hd], qOut[base:base+hd])
			copy(gate[h*hd:(h+1)*hd], qOut[base+hd:base+2*hd])
		}
	}
	// per-head RMSNorm on q/k
	for h := 0; h < nh; h++ {
		rmsNormVec(q[h*hd:(h+1)*hd], b.QNorm, 1e-6)
	}
	for h := 0; h < nkv; h++ {
		rmsNormVec(kOut[h*hd:(h+1)*hd], b.KNorm, 1e-6)
	}

	pos := b.KVOffset
	applyPartialRoPE(q, nh, hd, b.PartialRotary, b.RoPETheta, pos)
	applyPartialRoPE(kOut, nkv, hd, b.PartialRotary, b.RoPETheta, pos)

	// append KV
	need := (pos + 1) * kvDim
	if cap(b.KVCacheK) < need {
		nk := make([]float32, need)
		copy(nk, b.KVCacheK)
		b.KVCacheK = nk
		nv := make([]float32, need)
		copy(nv, b.KVCacheV)
		b.KVCacheV = nv
	} else {
		b.KVCacheK = b.KVCacheK[:need]
		b.KVCacheV = b.KVCacheV[:need]
	}
	copy(b.KVCacheK[pos*kvDim:], kOut)
	copy(b.KVCacheV[pos*kvDim:], vOut)
	b.KVOffset = pos + 1
	seq := b.KVOffset

	attnOut := make([]float32, qDim)
	scale := float32(1 / math.Sqrt(float64(hd)))
	rep := nh / nkv
	for h := 0; h < nh; h++ {
		kvH := h / rep
		qh := q[h*hd : (h+1)*hd]
		outH := attnOut[h*hd : (h+1)*hd]
		scores := make([]float32, seq)
		var maxS float32 = -1e30
		for t := 0; t < seq; t++ {
			kh := b.KVCacheK[t*kvDim+kvH*hd : t*kvDim+kvH*hd+hd]
			var s float32
			for d := 0; d < hd; d++ {
				s += qh[d] * kh[d]
			}
			s *= scale
			scores[t] = s
			if s > maxS {
				maxS = s
			}
		}
		var sum float32
		for t := 0; t < seq; t++ {
			scores[t] = float32(math.Exp(float64(scores[t] - maxS)))
			sum += scores[t]
		}
		inv := 1 / sum
		for d := 0; d < hd; d++ {
			var acc float32
			for t := 0; t < seq; t++ {
				vh := b.KVCacheV[t*kvDim+kvH*hd : t*kvDim+kvH*hd+hd]
				acc += scores[t] * inv * vh[d]
			}
			outH[d] = acc
		}
	}
	if b.OutputGate && gate != nil {
		for i := range attnOut {
			attnOut[i] *= 1 / (1 + float32(math.Exp(float64(-gate[i])))) // sigmoid
		}
	}
	return matVecDense(b.O, attnOut, y)
}

func matVecDense(l *dense.Layer, x, y []float32) error {
	if l == nil || l.Weights == nil {
		return fmt.Errorf("dense: nil")
	}
	if l.Exec.UseWebGPU || l.Exec.Backend == core.BackendWebGPU {
		in, out := l.Weights.Cols, l.Weights.Rows
		if l.Core.InputHeight > 0 {
			in = l.Core.InputHeight
		}
		if l.Core.OutputHeight > 0 {
			out = l.Core.OutputHeight
		}
		return dense.MatVecWebGPU(l.Weights, x, y, 1, in, out)
	}
	if l.Weights.Packed != nil {
		return dense.MatVecPackedBlob(l.Weights.Packed, x, y)
	}
	return weights.MatVec(l.Weights, x, y)
}

// matVecBlobGPU runs BinaryG128 (or any packed) GEMV on WebGPU when requested.
func matVecBlob(b *quant.Blob, x, y []float32, useGPU bool) error {
	if b == nil {
		return fmt.Errorf("nil blob")
	}
	if useGPU {
		ws, err := weights.FromBlob(b)
		if err != nil {
			return err
		}
		return dense.MatVecWebGPU(ws, x, y, 1, b.Cols, b.Rows)
	}
	return dense.MatVecPackedBlob(b, x, y)
}

func rmsNormVec(x, gamma []float32, eps float64) {
	var mean float64
	for _, v := range x {
		mean += float64(v) * float64(v)
	}
	mean /= float64(len(x))
	inv := float32(1 / math.Sqrt(mean+eps))
	for i := range x {
		g := float32(1)
		if i < len(gamma) {
			g = gamma[i]
		}
		x[i] = x[i] * inv * g
	}
}

func applyPartialRoPE(x []float32, nHeads, headDim int, partial, theta float64, pos int) {
	rotDim := int(float64(headDim) * partial)
	if rotDim <= 0 {
		rotDim = headDim
	}
	if rotDim%2 != 0 {
		rotDim--
	}
	for h := 0; h < nHeads; h++ {
		base := h * headDim
		for i := 0; i < rotDim; i += 2 {
			freq := 1 / math.Pow(theta, float64(i)/float64(rotDim))
			angle := float64(pos) * freq
			cos := float32(math.Cos(angle))
			sin := float32(math.Sin(angle))
			u, v := x[base+i], x[base+i+1]
			x[base+i] = u*cos - v*sin
			x[base+i+1] = u*sin + v*cos
		}
	}
}
