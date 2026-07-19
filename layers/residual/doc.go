// Package residual is Welvet Residual block (loom y = F(x) + x).
//
// F is an ordered list of Dense Dim→Dim children (same packing as Sequential).
// Forward: fx = F(x); y = fx + x. Backward: gradIn = ∂F/∂x + ∂L/∂y.
//
// Contract: CPU tiled + SIMD + WebGPU via child Dense. No QAT. Tests in w2a.
package residual
