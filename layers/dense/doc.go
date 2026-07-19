// Package dense is the Welvet fully-connected layer.
//
// Contract: CPU tiled, Plan 9 SIMD, and WebGPU — each path is exact.
// Activations/grads are core.Tensor[T] for any core.Numeric (not hardcoded float32).
// Weight DType matrix is all 34 core.AllDTypes (0–33) on CPU FormatNone.
// Unimplemented dtype×quant×backend combinations return errors (no fallback).
// Tests live in github.com/openfluke/w2a only.
package dense
