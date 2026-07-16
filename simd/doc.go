// Package simd provides Plan 9 assembly kernels for Welvet:
//
//   - amd64: AVX2+FMA (dotF32AccF64Avx2, Q4, int8/uint8, saxpy, BitNet ternary)
//   - arm64: NEON (dot, Q4, int8/uint8, saxpy, BitNet ternary/TL1)
//   - other GOARCH: simdEnabled()==false — BackendSIMD must hard-error (no soft fallback)
//
// Public entry points: DotTile, DotF32, DotI8Tile, DotU8Tile, DotQ4_0Row,
// SaxpyF32AccF64, BitNet* helpers. Dense/SIMD backend calls these directly.
package simd
