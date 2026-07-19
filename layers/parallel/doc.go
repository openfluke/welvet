// Package parallel is Parallel / MoE combine (loom Parallel).
//
// Branches are Dense children. Combine modes: concat (default), add, avg, filter (MoE gate).
// Contract: CPU tiled + SIMD + WebGPU via children Exec; dtype × k-quant on branches/gate.
// No QAT. Tests live in github.com/openfluke/w2a — not here.
package parallel
