// Package nativemlx binds welvet.mlx (Apple MLX / Metal GPU) via C ABI.
//
// Heterogeneous Welvet flow (user-owned sync):
//
//	1. Build the network in Go (weights live in Go Stores).
//	2. SyncToMLX on a layer (or rebuild a Sequential) — pushes W into the GPU path.
//	3. Set Exec.Backend = core.BackendMLX (or call ForwardMLX).
//	4. Forward pivots Go / nativecpp / MLX per layer.
//	5. SyncFromMLX when you want Go Stores to mirror GPU-side buffers again
//	   (forward-only today does not mutate W; sync-back is for API symmetry / future).
//
// Caps: LayerSupported(name, dtype) / SupportedLayers() — Offload rejects unsupported mounts.
//
// Platform: darwin/arm64 + cgo + shipped native/mlx/darwin_arm64 + Metal GPU.
// Other builds stub. Available() requires Metal GPU (no CPU fallback).
package nativemlx
