# darwin_arm64

Shipped `libwelvet` for macOS Apple Silicon.

| File | Role |
|------|------|
| `libwelvet.dylib` | Shared library (runtime + cgo `-lwelvet`) |

Build from welvet.cpp (engine pin in `../MANIFEST.txt`):

```bash
cd welvet.cpp
./scripts/build.sh
strip -x -o ../welvet/native/cpp/darwin_arm64/libwelvet.dylib build/libwelvet.dylib
```

Go binding: `nativecpp` with `cgo && darwin && arm64`. Rpath is set to this directory.
Set `WELVET_DENSE_FWD` to `welvet.cpp/build/welvet_dense_fwd` for go_v_cpp tools.
