// Package mamba is a selective SSM mixer (seqmix.KindSSM).
//
// Minimal Mamba-style block: InProj → softplus(Δ) selective scan → OutProj.
// Contract: CPU tiled + SIMD + WebGPU (host scan + Dense MatVec), dtype × k-quant.
// No QAT. Tests live in github.com/openfluke/w2a — not here.
package mamba
