// Package nativeqnn binds welvet.qnn (Qualcomm Hexagon NPU / QNN HTP) via C ABI.
//
// Heterogeneous Welvet flow (user-owned sync):
//
//	1. Build the network in Go (weights live in Go Stores).
//	2. SyncToQNN on a layer (or rebuild a Sequential) — pushes W into the NPU path.
//	3. Set Exec.Backend = core.BackendQNN (or call ForwardQNN).
//	4. Forward pivots Go / nativecpp / QNN per layer.
//	5. SyncFromQNN when you want Go Stores to mirror NPU-side buffers again
//	   (forward-only today does not mutate W; sync-back is for API symmetry / future).
//
// Caps: LayerSupported(name, dtype) / SupportedLayers() — Offload rejects unsupported mounts.
//
// Platform: windows/arm64 + cgo + shipped native/qnn/windows_arm64. Other builds stub.
package nativeqnn
