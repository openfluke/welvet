# windows_amd64

Shipped `libwelvet` for Windows x86-64 (llvm-mingw / gnullvm-compatible).

| File | Role |
|------|------|
| `libwelvet.dll` | Shared library (runtime) |
| `libwelvet.dll.a` | Import lib for cgo `-lwelvet` |

Built from welvet.cpp **0.67** (`scripts/build.sh` / cmake + llvm-mingw).

Go binding: `nativecpp` with `cgo && windows && amd64`. Put this directory on `PATH` (or next to the test binary) so the DLL loads at runtime.

Sign-off: `w2a/scripts/run_go_v_cpp_windows_snapshot.cmd` → `logs/windowsx86.txt`.
