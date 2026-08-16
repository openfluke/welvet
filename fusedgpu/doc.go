// Package fusedgpu runs a full decoder on WebGPU:
//   - Q4_0 (Lucy-style) via Engine / NewFromSpec
//   - BinaryG128 hybrid (Qwen3.5 / Bonsai) via HybridEngine / NewHybridFromSpec
//   - Android native Vulkan hybrid via HybridVKEngine / NewHybridVKFromSpec
//
// Throughput on Adreno (see notes/rnd_qual.md):
//  1. Wave-1 FA fusion (bingemv_qkv_rms / attn_prep / binswiglu_rms) cuts
//     full-attention decode from ~285 → ~173 disp/tok (6/layer + overhead).
//  2. AOT SPIR-V (scripts/compile-hybrid-spirv.sh → fusedgpu/spirv/*.spv) skips
//     runtime WGSL→SPIR-V translation when blobs are embedded.
//  3. Native Vulkan (default on Android builds; WELVET_NATIVE_VK=0 or UI chip
//     for WebGPU A/B) bypasses Dawn’s barrier tax; full_attention-only for now.
//
// Hybrid fuse keeps every weight + GDN/attn/FFN scratch on device; host only
// supplies token IDs and reads back logits. Needs enough VRAM for the full entity.
package fusedgpu
