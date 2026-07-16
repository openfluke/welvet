// Package fusedgpu runs a full decoder on WebGPU:
//   - Q4_0 (Lucy-style) via Engine / NewFromSpec
//   - BinaryG128 hybrid (Qwen3.5 / Bonsai) via HybridEngine / NewHybridFromSpec
//
// Hybrid fuse keeps every weight + GDN/attn/FFN scratch on device; host only
// supplies token IDs and reads back logits. Needs enough VRAM for the full entity.
package fusedgpu
