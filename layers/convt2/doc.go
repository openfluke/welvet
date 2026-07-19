// Package convt2 is ConvTranspose2d (loom ConvTransposed2D).
//
// Weights live on Proj (Dense Filters × InChannels·Kernel²), same layout as cnn2.
// Contract: CPU tiled + SIMD + WebGPU, dtype × k-quant. No QAT.
// Tests live in github.com/openfluke/w2a — not here.
package convt2
