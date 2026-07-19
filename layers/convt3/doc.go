// Package convt3 is ConvTranspose3d (loom ConvTransposed3D).
//
// Weights live on Proj (Dense Filters × InChannels·Kernel³), same layout as cnn3.
// Contract: CPU tiled + SIMD + WebGPU, dtype × k-quant. No QAT.
// Tests live in github.com/openfluke/w2a — not here.
package convt3
