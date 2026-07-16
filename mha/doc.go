// Package mha is Welvet multi-head attention — the KindAttention seqmix.
//
// Policy axes (future-proof for transformers, diffusion, Prefix-LM, local attn):
//   - Mask: causal | bidirectional | sliding_window | prefix_lm | custom
//   - Pos:  RoPE | none | ALiBi | RoPE+ALiBi
//   - Mode: self | cross (K/V from Context — encoder / diffusion cond)
//   - Softmax, QK-RMSNorm, GQA/MQA (NumKVHeads), optional ScaleOverride
//
// Presets: DecoderCausal, EncoderBidirectional, CrossAttention,
// DiffusionSelf, DiffusionCross, PrefixLMAttn, LocalCausal, ALiBiCausal.
//
// Contract: Q/K/V/O projections ride dense.Layer (FormatNone×34 + all quants ×
// CPU/SIMD/WebGPU). Attention ALU is f64-accurate host on every backend
// (projections still use the selected backend — no silent MatVec fallback).
// Activations are core.Tensor[T] (not hardcoded float32). No QAT.
// Tests live in github.com/openfluke/w2a only.
//
// Non-attention mixers (Mamba/SSM, linear attn, Hyena) live in their own
// packages under the seqmix contract — not as silent forks of this file.
package mha
