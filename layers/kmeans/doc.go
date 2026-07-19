// Package kmeans is differentiable soft K-Means (loom KMeans).
//
// Centers live on a Dense store (K × FeatureDim). Output modes: probabilities | features.
// Contract: CPU tiled + SIMD + WebGPU, dtype × k-quant. No QAT.
// Tests live in github.com/openfluke/w2a — not here.
package kmeans
