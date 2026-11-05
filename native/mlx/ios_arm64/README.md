# ios_arm64

**Not shipped in the Go module** — Apple MLX iPhoneOS artifacts are too large:

| Artifact | Approx size |
|----------|-------------|
| `mlx.metallib` | ~140MB |
| `libmlx.a` | ~11MB |
| `libwelvet_mlx.a` | ~250KB |

Build locally (needs Xcode MetalToolchain):

```bash
xcodebuild -downloadComponent MetalToolchain   # once
xcrun metal -v
cd ../welvet.mlx
./scripts/ios_install_go.sh
# → writes libwelvet_mlx.a + libmlx.a + mlx.metallib here (gitignored)
```

Link with `-force_load` plus `-framework Metal -framework Foundation -framework Accelerate`,
and bundle `mlx.metallib`.

For on-device Apple GPU in-repo today, use **`native/metal/ios_arm64/`** (`libwelvet_metal.a` is small and shipped).
Go `nativemlx` cgo remains macOS (`darwin_arm64` dylib).
