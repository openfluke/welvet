# windows_arm64 — welvet.qnn (Hexagon NPU)

Shipped `libwelvet_qnn` for Windows ARM64 (llvm-mingw). Forward-only QNN HTP.

| File | Role |
|------|------|
| `libwelvet_qnn.dll` | Runtime (put on `PATH` or next to the test binary) |
| `libwelvet_qnn.dll.a` / `libwelvet_qnn.a` | Import lib for cgo (`-lwelvet_qnn`; `.a` is a copy so Go `-static` does not pull a full C++ archive) |
| `../include/welvet_qnn_c.h` | C ABI |

Built from `welvet.qnn` **1.0.0**. Needs QAIRT on the machine (`QAIRT_SDK_ROOT`), Memory Integrity **OFF** for HTP create, and llvm-mingw runtime (`libc++.dll` / `libunwind.dll`) on `PATH` next to this DLL.

Go binding: package `nativeqnn` (`cgo && windows && arm64`).
