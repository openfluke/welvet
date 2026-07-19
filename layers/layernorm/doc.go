// Package layernorm is Welvet LayerNorm (loom / Transformer semantics).
//
// Per-token over last dim: y = ((x−μ)/√(var+ε))⊙γ + β.
// Affine γ,β live on weights.Store (1×Dim each). FormatNone×34 + all quants
// × CPU/SIMD/WebGPU via the same permutation matrix as Dense.
//
// Contract: CPU tiled + SIMD + WebGPU, native dtype × k-quant forward/backward.
// No QAT. Tests/docs/CABI live in github.com/openfluke/w2a — not here.
package layernorm
