// Package rmsnorm is Welvet RMSNorm (Llama / loom semantics).
//
// Math (per token / last dim):
//
//	rms = sqrt(mean(x²) + eps)
//	y   = (x / rms) ⊙ γ
//
// γ lives in weights.Store (FormatNone×34 + all quants × CPU/SIMD/WebGPU decode).
// Norm ALU is f64-accurate host; SIMD uses DotTile for Σx² when BackendSIMD.
// Activations are core.Tensor[T] (not hardcoded float32). No QAT.
// Tests live in github.com/openfluke/w2a only.
package rmsnorm
