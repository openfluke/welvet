// Package convt1 is ConvTranspose1d (loom ConvTransposed1D).
//
// Weights live on Proj (Dense Filters × InChannels·Kernel), same layout as cnn1.
// Forward scatters input × kernel into an upsampled map; backward is the adjoint.
// Contract: CPU tiled + SIMD + WebGPU (host layout + Dense MatVec), dtype × k-quant.
// No QAT. Tests live in github.com/openfluke/w2a — not here.
package convt1
