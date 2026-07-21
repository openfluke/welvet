// Package gdn is Gated DeltaNet (Qwen3.5 / Bonsai linear_attention; seqmix.KindLinearAttn).
//
// Inference decode path is primary (ForwardDecode). Tensor Forward loops decode over T.
// Exec.Backend dispatches CPU / SIMD / WebGPU (hard error if SIMD off or no device).
// WebGPU stages projection GEMVs on device when Available(); decode ALU stays host.
// Backward (train.go) is a practical, truncated-BPTT training path: exact grads for
// Out/NormGamma/InZ and for the InQKV/InB/InA linear maps' current-token contribution;
// the recurrent State (and conv history) is treated as stop-gradient across tokens —
// not full loom-parity BPTT through time, but not a silent zero grad either.
// Tests live in github.com/openfluke/w2a — not here.
package gdn
