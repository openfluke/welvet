package flux2

import "math"

// JointAttention runs Flux2 double-stream joint attention.
//
// Layout conventions (batch=1):
//   q/k/vImg: [imgSeq * heads * headDim] as [imgSeq][heads][headDim]
//   q/k/vTxt: [txtSeq * heads * headDim]
// Output attnImg / attnTxt: [seq * (heads*headDim)] flattened last two dims.
func JointAttention(
	qImg, kImg, vImg []float32,
	qTxt, kTxt, vTxt []float32,
	normQ, normK, normAddedQ, normAddedK []float32,
	rope RotaryEmb,
	imgSeq, txtSeq, heads, headDim int,
	eps float64,
	attnImg, attnTxt []float32,
) {
	inner := heads * headDim
	total := txtSeq + imgSeq

	// QK RMSNorm per head (elementwise affine on headDim)
	rmsHeads(qImg, normQ, imgSeq, heads, headDim, eps)
	rmsHeads(kImg, normK, imgSeq, heads, headDim, eps)
	rmsHeads(qTxt, normAddedQ, txtSeq, heads, headDim, eps)
	rmsHeads(kTxt, normAddedK, txtSeq, heads, headDim, eps)

	// Concat txt||img along sequence → [total, heads, headDim]
	q := make([]float32, total*inner)
	k := make([]float32, total*inner)
	v := make([]float32, total*inner)
	copy(q, qTxt)
	copy(k, kTxt)
	copy(v, vTxt)
	copy(q[txtSeq*inner:], qImg)
	copy(k[txtSeq*inner:], kImg)
	copy(v[txtSeq*inner:], vImg)

	ApplyRotaryEmb(q, rope, total, heads, headDim)
	ApplyRotaryEmb(k, rope, total, heads, headDim)

	out := make([]float32, total*inner)
	scaledDotProductAttention(q, k, v, out, total, heads, headDim)

	copy(attnTxt, out[:txtSeq*inner])
	copy(attnImg, out[txtSeq*inner:])
}

// SelfAttention runs single-stream attention (no txt split) with QK RMSNorm + RoPE.
func SelfAttention(
	q, k, v []float32,
	normQ, normK []float32,
	rope RotaryEmb,
	seq, heads, headDim int,
	eps float64,
	out []float32,
) {
	rmsHeads(q, normQ, seq, heads, headDim, eps)
	rmsHeads(k, normK, seq, heads, headDim, eps)
	ApplyRotaryEmb(q, rope, seq, heads, headDim)
	ApplyRotaryEmb(k, rope, seq, heads, headDim)
	scaledDotProductAttention(q, k, v, out, seq, heads, headDim)
}

func rmsHeads(x, weight []float32, seq, heads, headDim int, eps float64) {
	for s := 0; s < seq; s++ {
		for h := 0; h < heads; h++ {
			off := (s*heads + h) * headDim
			RMSNorm(x[off:off+headDim], weight, headDim, eps)
		}
	}
}

// scaledDotProductAttention: q/k/v/out as [seq][heads][headDim].
func scaledDotProductAttention(q, k, v, out []float32, seq, heads, headDim int) {
	scale := 1.0 / math.Sqrt(float64(headDim))
	scores := make([]float64, seq)
	for h := 0; h < heads; h++ {
		for qi := 0; qi < seq; qi++ {
			qOff := (qi*heads + h) * headDim
			maxScore := math.Inf(-1)
			for kj := 0; kj < seq; kj++ {
				kOff := (kj*heads + h) * headDim
				var dot float64
				for d := 0; d < headDim; d++ {
					dot += float64(q[qOff+d]) * float64(k[kOff+d])
				}
				sc := dot * scale
				scores[kj] = sc
				if sc > maxScore {
					maxScore = sc
				}
			}
			var sum float64
			for kj := 0; kj < seq; kj++ {
				scores[kj] = math.Exp(scores[kj] - maxScore)
				sum += scores[kj]
			}
			inv := 1.0 / sum
			oOff := (qi*heads + h) * headDim
			for d := 0; d < headDim; d++ {
				var acc float64
				for kj := 0; kj < seq; kj++ {
					vOff := (kj*heads+h)*headDim + d
					acc += scores[kj] * inv * float64(v[vOff])
				}
				out[oOff+d] = float32(acc)
			}
		}
	}
}
