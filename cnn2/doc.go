// Package cnn2 is Welvet 2D convolution (loom Conv2d semantics).
//
// Layout: input/output [batch, channels, height, width]; weights
// [filters × in × k × k] via a Dense projection over im2col patches —
// same FormatNone×34 + all quants × CPU/SIMD/WebGPU coverage as Dense.
//
// Contract: CPU tiled + SIMD + WebGPU, native dtype × k-quant forward/backward.
// No QAT. Tests/docs/CABI live in github.com/openfluke/w2a — not here.
package cnn2
