// Package fusedgpu runs a full Q4_0 decoder on WebGPU (Lucy-style single-pass/token).
//
// Requires a welvet transformer.Model with baked FormatQ4_0 weights and lm_head.packed.
package fusedgpu
