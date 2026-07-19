// Package seqmix is the Welvet sequence-mixing contract.
//
// Every token→token mixer (attention, SSM/Mamba, linear attention, Hyena, …)
// follows the same dtype × quant × backend × fwd/bwd matrix rules as Dense/MHA.
// Concrete packages implement the op; this package only names kinds + documents
// the shared Forward/Backward shape so transformers, diffusion, and SSM stacks
// can swap mixers without rewriting the volumetric walk.
//
//	mha/     KindAttention  (done)
//	mamba/   KindSSM        (CPU selective scan)
//	gdn/     KindLinearAttn (decode / gated delta)
//	…        KindConvMix    (future)
//
// Tests live in github.com/openfluke/w2a — not here.
package seqmix

import "fmt"

// Kind identifies which sequence mixer a cell uses.
type Kind int

const (
	KindAttention  Kind = iota // MHA / GQA / MQA / cross / local / ALiBi …
	KindSSM                    // Mamba / S6 / related state-space
	KindLinearAttn             // Performer / Linear Transformer / RetNet / GDN-style
	KindConvMix                // Hyena / FFT conv mixers
)

func (k Kind) String() string {
	switch k {
	case KindAttention:
		return "attention"
	case KindSSM:
		return "ssm"
	case KindLinearAttn:
		return "linear_attn"
	case KindConvMix:
		return "conv_mix"
	default:
		return fmt.Sprintf("Kind(%d)", int(k))
	}
}

// Contract documents the shared I/O shape for sequence mixers.
// Implementors expose package-level Forward/Backward matching Dense:
//
//	Forward(l, x[B,T,D]) → (pre, post[B,T,D])
//	Backward(l, gy, x, pre) → (gx, dW)
//
// State (SSM/GDN) is owned by the layer; call Reset between independent sequences.
type Contract struct {
	Kind   Kind
	DModel int
	MaxT   int
}
