# linux_arm64

Shipped `libwelvet` for Linux aarch64 (Fedora/WSL or native).

| File | Role |
|------|------|
| `libwelvet.so` | Shared library (runtime + cgo `-lwelvet`) |

Build from welvet.cpp (engine pin in `../MANIFEST.txt`):

```bash
cd welvet.cpp
cmake -S . -B build-linux-arm64 -DCMAKE_BUILD_TYPE=Release
cmake --build build-linux-arm64 -j"$(nproc)"
strip -o ../welvet/native/cpp/linux_arm64/libwelvet.so build-linux-arm64/libwelvet.so
```

Go binding: `nativecpp` with `cgo && linux && arm64`. Rpath is set to this directory.
Set `WELVET_DENSE_FWD` to `welvet.cpp/build-linux-arm64/welvet_dense_fwd` for go_v_cpp tools.
