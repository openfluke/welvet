package wav2vec2

import (
	"fmt"
	"math"
)

type linearLayer struct {
	W, B []float32
	In, Out int
}

type layerNormParams struct {
	W, B []float32
}

type convLayer struct {
	W              []float32 // [out][in][k]
	OutC, InC, K   int
	Stride         int
	NormW, NormB   []float32 // optional GroupNorm (layer 0)
	HasGroupNorm   bool
}

type encoderLayer struct {
	Q, K, V, Out linearLayer
	AttnNorm     layerNormParams
	FFInter      linearLayer
	FFOut        linearLayer
	FinalNorm    layerNormParams
}

// Model is Wav2Vec2ForCTC (base-960h layout).
type Model struct {
	Cfg   Config
	Vocab *Vocab

	Feats []convLayer

	ProjNorm layerNormParams
	Proj     linearLayer

	PosW, PosB []float32 // fused weight_norm conv weight [768][48][128], bias
	PosK       int
	PosGroups  int
	PosInPerG  int

	EncNorm layerNormParams
	Layers  []encoderLayer

	LMHead linearLayer
}

// ForwardPCM runs waveform → logits [T, vocab].
func (m *Model) ForwardPCM(waveform []float32) ([]float32, int, error) {
	if m == nil {
		return nil, 0, fmt.Errorf("wav2vec2: nil model")
	}
	if len(waveform) == 0 {
		return nil, 0, fmt.Errorf("wav2vec2: empty waveform")
	}

	// Feature extractor: start as [1][T]
	cur := append([]float32(nil), waveform...)
	inC, inT := 1, len(waveform)
	for i, cl := range m.Feats {
		out, outT := conv1dValid(cl.W, cl.OutC, cl.InC, cl.K, cl.Stride, cur, inT)
		if outT == 0 {
			return nil, 0, fmt.Errorf("wav2vec2: feature conv %d produced empty seq (inT=%d)", i, inT)
		}
		if cl.HasGroupNorm {
			groupNormPerChannel(out, cl.OutC, outT, cl.NormW, cl.NormB, m.Cfg.LayerNormEps)
		}
		for j := range out {
			out[j] = gelu(out[j])
		}
		cur, inC, inT = out, cl.OutC, outT
	}

	// [C,T] → [T,C] for projection
	hidden := transposeCHWToTHC(cur, inC, inT)
	for t := 0; t < inT; t++ {
		layerNorm(hidden[t*inC:(t+1)*inC], m.ProjNorm.W, m.ProjNorm.B, m.Cfg.LayerNormEps)
	}
	hidden = linear(hidden, m.Proj.W, m.Proj.B, inT, m.Proj.In, m.Proj.Out)
	// hidden [T, H]

	// Convolutional positional embedding (channels-first for conv).
	hCHW := transposeTHCToCHW(hidden, inT, m.Cfg.HiddenSize)
	pos := conv1dGrouped(m.PosW, m.PosB, m.Cfg.HiddenSize, m.PosInPerG, m.PosK, m.PosGroups, m.PosK/2, hCHW, inT, true)
	posT := len(pos) / m.Cfg.HiddenSize
	if posT != inT {
		// HF keeps same length after pad+trim; enforce.
		if posT < inT {
			return nil, 0, fmt.Errorf("wav2vec2: pos embed len %d < %d", posT, inT)
		}
		// trim extra if any
		trimmed := make([]float32, m.Cfg.HiddenSize*inT)
		for c := 0; c < m.Cfg.HiddenSize; c++ {
			copy(trimmed[c*inT:(c+1)*inT], pos[c*posT:c*posT+inT])
		}
		pos = trimmed
	}
	posTHC := transposeCHWToTHC(pos, m.Cfg.HiddenSize, inT)
	addInPlace(hidden, posTHC)

	for t := 0; t < inT; t++ {
		layerNorm(hidden[t*m.Cfg.HiddenSize:(t+1)*m.Cfg.HiddenSize], m.EncNorm.W, m.EncNorm.B, m.Cfg.LayerNormEps)
	}

	for li := range m.Layers {
		var err error
		hidden, err = m.Layers[li].forward(hidden, inT, m.Cfg)
		if err != nil {
			return nil, 0, err
		}
	}

	logits := linear(hidden, m.LMHead.W, m.LMHead.B, inT, m.LMHead.In, m.LMHead.Out)
	return logits, inT, nil
}

func (l *encoderLayer) forward(x []float32, t int, cfg Config) ([]float32, error) {
	h := cfg.HiddenSize
	residual := append([]float32(nil), x...)
	attnOut, err := multiHeadSelfAttention(x, t, cfg.NumAttentionHeads, cfg.headDim(), l.Q, l.K, l.V, l.Out)
	if err != nil {
		return nil, err
	}
	addInPlace(attnOut, residual)
	for i := 0; i < t; i++ {
		layerNorm(attnOut[i*h:(i+1)*h], l.AttnNorm.W, l.AttnNorm.B, cfg.LayerNormEps)
	}

	residual = append([]float32(nil), attnOut...)
	ff := linear(attnOut, l.FFInter.W, l.FFInter.B, t, l.FFInter.In, l.FFInter.Out)
	for i := range ff {
		ff[i] = gelu(ff[i])
	}
	ff = linear(ff, l.FFOut.W, l.FFOut.B, t, l.FFOut.In, l.FFOut.Out)
	addInPlace(ff, residual)
	for i := 0; i < t; i++ {
		layerNorm(ff[i*h:(i+1)*h], l.FinalNorm.W, l.FinalNorm.B, cfg.LayerNormEps)
	}
	return ff, nil
}

func multiHeadSelfAttention(x []float32, t, heads, headDim int, q, k, v, o linearLayer) ([]float32, error) {
	h := heads * headDim
	if q.In != h || len(x) != t*h {
		return nil, fmt.Errorf("wav2vec2: attn shape")
	}
	Q := linear(x, q.W, q.B, t, q.In, q.Out)
	K := linear(x, k.W, k.B, t, k.In, k.Out)
	V := linear(x, v.W, v.B, t, v.In, v.Out)
	scale := float32(1 / math.Sqrt(float64(headDim)))

	ctx := make([]float32, t*h)
	scores := make([]float32, t)
	for head := 0; head < heads; head++ {
		off := head * headDim
		for qi := 0; qi < t; qi++ {
			qRow := Q[qi*h+off : qi*h+off+headDim]
			maxScore := float32(-math.MaxFloat32)
			for kj := 0; kj < t; kj++ {
				kRow := K[kj*h+off : kj*h+off+headDim]
				var dot float32
				for d := 0; d < headDim; d++ {
					dot += qRow[d] * kRow[d]
				}
				s := dot * scale
				scores[kj] = s
				if s > maxScore {
					maxScore = s
				}
			}
			var sum float64
			for kj := 0; kj < t; kj++ {
				scores[kj] = float32(math.Exp(float64(scores[kj] - maxScore)))
				sum += float64(scores[kj])
			}
			inv := float32(1 / sum)
			outRow := ctx[qi*h+off : qi*h+off+headDim]
			for d := range outRow {
				outRow[d] = 0
			}
			for kj := 0; kj < t; kj++ {
				a := scores[kj] * inv
				vRow := V[kj*h+off : kj*h+off+headDim]
				for d := 0; d < headDim; d++ {
					outRow[d] += a * vRow[d]
				}
			}
		}
	}
	return linear(ctx, o.W, o.B, t, o.In, o.Out), nil
}

// TranscribePCM runs forward + greedy CTC decode.
func (m *Model) TranscribePCM(waveform []float32) (string, error) {
	logits, t, err := m.ForwardPCM(waveform)
	if err != nil {
		return "", err
	}
	vsz := m.LMHead.Out
	ids := make([]int, t)
	for i := 0; i < t; i++ {
		base := i * vsz
		best := 0
		bestV := logits[base]
		for c := 1; c < vsz; c++ {
			if logits[base+c] > bestV {
				bestV = logits[base+c]
				best = c
			}
		}
		ids[i] = best
	}
	return m.Vocab.DecodeCTCGreedy(ids), nil
}

// TranscribeFile loads/normalizes a WAV and transcribes.
func (m *Model) TranscribeFile(path string) (string, error) {
	x, err := PrepareAudio(path)
	if err != nil {
		return "", err
	}
	return m.TranscribePCM(x)
}
