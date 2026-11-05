# nativemlx — Apple MLX GPU C ABI

Go bindings for `libwelvet_mlx` (welvet.mlx). Forward-only Metal GPU; no CPU fallback.

| Build | Behavior |
|-------|----------|
| `cgo && darwin && arm64` | Links `native/mlx/darwin_arm64/libwelvet_mlx.dylib` |
| other | Stub (`Available() == false`) |

Install the dylib from the sibling engine:

```bash
../welvet.mlx/scripts/macos_install_go.sh
```

`Available()` is true only when the library loads **and** Metal GPU detect succeeds.
