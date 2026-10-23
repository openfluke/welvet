# native/qnn — Hexagon NPU (welvet.qnn)

Shipped `libwelvet_qnn` for Welvet Go `nativeqnn` (cgo).

| Path | Platform |
|------|----------|
| `windows_arm64/` | Snapdragon X / Windows ARM64 |
| `include/welvet_qnn_c.h` | C ABI |

## User-owned weight sync

Weights live in Go `weights.Store` until you push them:

```text
1. Build network in Go (default BackendCPUTiled)
2. layer.SyncToQNN()          // caps-check + copy W → QNN mirror
3. layer.Exec.Backend = core.BackendQNN
4. Forward…                  // HTP
5. layer.SyncFromQNN()       // optional: mirror → Go Stores
```

Unsupported mounts (`RequireLayer` / `LayerSupported`) **reject** — see `nativeqnn.SupportedLayers()`.

Heterogeneous Sequential: leave parent on `BackendCPUTiled`, SyncToQNN + `BackendQNN` on individual children (Dense / RMSNorm / LayerNorm / Softmax). Or SyncToQNN on the Sequential and set parent `BackendQNN` for a fused stack.

Runtime: put `windows_arm64` on `PATH` (or next to the binary). Needs QAIRT + Memory Integrity **OFF** for HTP create.
