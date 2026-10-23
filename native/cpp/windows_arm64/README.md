# windows_arm64

Shipped `libwelvet` for Windows ARM64 (llvm-mingw / gnullvm-compatible).

| File | Role |
|------|------|
| `libwelvet.dll` | Shared library (runtime) |
| `libwelvet.dll.a` / `libwelvet.a` | Import lib for cgo `-lwelvet` |

Built from welvet.cpp **0.67** (`scripts/build.sh` / cmake + llvm-mingw).

Go binding: `nativecpp` with `cgo && windows && arm64`. Put this directory on `PATH` (or next to the test binary) so the DLL loads at runtime.
