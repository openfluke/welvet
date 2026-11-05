# native/mlx — Apple MLX GPU (welvet.mlx)

Shipped `libwelvet_mlx` for Welvet Go `nativemlx` (cgo).

| Path | Platform |
|------|----------|
| `darwin_arm64/` | Apple silicon macOS |
| `include/welvet_mlx_c.h` | C ABI |

Install from the sibling engine checkout:

```bash
../welvet.mlx/scripts/macos_install_go.sh
```

## User-owned weight sync

Weights live in Go `weights.Store` until you push them:

```text
1. Build network in Go (default BackendCPUTiled)
2. layer.SyncToMLX()          // caps-check + copy W → MLX mirror
3. layer.Exec.Backend = core.BackendMLX
4. Forward…                  // Metal GPU via MLX
5. layer.SyncFromMLX()       // optional: mirror → Go Stores
```

Unsupported mounts (`RequireLayer` / `LayerSupported`) **reject** — see `nativemlx.SupportedLayers()`.

Runtime: Apple silicon + Metal GPU. Host buffers stay float32; `dtype` selects MLX compute type. Forward-only (no backward).
