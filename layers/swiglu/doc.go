// Package swiglu is Welvet SwiGLU FFN (SiLU-gated MLP).
//
// Math (loom / planetbridging compatible):
//
//	gate = Wg x + bg
//	up   = Wu x + bu
//	h    = SiLU(gate) ⊙ up
//	y    = Wd h + bd
//
// Gate / Up / Down ride dense.Layer (FormatNone×34 + all quants × CPU/SIMD/WebGPU).
// Activations are core.Tensor[T] (not hardcoded float32). No QAT.
// Tests live in github.com/openfluke/w2a only.
package swiglu
