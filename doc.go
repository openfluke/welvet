// Package welvet is a Golang AI engine for local neural network training,
// inference, quantization, and GPU/NPU acceleration.
//
// Welvet provides neural network layers (Dense, Multi-Head Attention, CNN,
// RNN/LSTM, SwiGLU, normalization layers), 34 numerical data types, native
// quantization (including k-quants), and multi-backend hardware execution:
//
//   - CPU: Portable tiled Go, Plan 9 SIMD, and NativeCPU (welvet.cpp)
//   - GPU: NVIDIA CUDA, AMD ROCm/HIP, Apple Metal, Apple MLX, Vulkan, and WebGPU
//   - NPU: Qualcomm Hexagon HTP (QNN)
//
// # Core Subpackages
//
// Import specific subpackages to build and run models:
//
//	import (
//		"github.com/openfluke/welvet/core"
//		"github.com/openfluke/welvet/weights"
//		"github.com/openfluke/welvet/layers/dense"
//	)
//
// For full documentation, benchmarks, and guides, visit https://openfluke.com.
package welvet
