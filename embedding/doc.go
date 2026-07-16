// Package embedding is Welvet token embedding (loom gather/scatter).
//
// Layout: input token IDs [batch, seq] (or [seq]); output [batch, seq, emb].
// Weights: row-major vocab × emb in a weights.Store (FormatNone×34 + all quants).
// Forward gathers rows; backward scatters dW (gradIn is zeros — discrete IDs).
//
// Not Dense MatVec — host gather/scatter ALU. SIMD requires Plan 9 enabled;
// WebGPU requires a real device then host ALU (no fused gather shader yet).
//
// Contract: CPU tiled + SIMD + WebGPU. No QAT. Tests live in w2a.
package embedding
