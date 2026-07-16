// Package softmax is Welvet Softmax (loom weightless activation).
//
// No learnable weights — host max-subtract Softmax over the last axis (or
// explicit Grid rows×cols). Temperature scales logits; backward applies
// Jacobian diag(y)−yyᵀ and ×(1/T).
//
// pre and post are both probabilities y (needed for BP Jacobian on the tape).
// FormatNone/quant axes in w2a exercise ALU only (no weight store).
//
// Contract: CPU tiled + SIMD + WebGPU. No QAT. Tests live in w2a.
package softmax
