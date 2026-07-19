// Package gdn is Gated DeltaNet (Qwen3.5 / Bonsai linear_attention; seqmix.KindLinearAttn).
//
// Inference decode path is primary (ForwardDecode). Tensor Forward loops decode over T.
// Training/backward is not yet loom-parity (load-only + smoke suites).
// Tests live in github.com/openfluke/w2a — not here.
package gdn
