# windows_amd64

Shipped `libwelvet_cuda` for Windows x86-64 (MSVC + NVIDIA CUDA Toolkit).

| File | Role |
|------|------|
| `libwelvet_cuda.dll` | Shared library (runtime) |
| `libwelvet_cuda.lib` | Import lib (optional; Go loads the DLL dynamically) |

Built from welvet.cuda **0.79** (`scripts/build_windows.cmd` / cmake + MSVC + nvcc).

Go binding: `nativecuda` with `cgo && windows && amd64`. Put this directory (or CUDA `bin`) on `PATH` so `cudart` resolves at runtime, or set `WELVET_CUDA_LIBRARY` to the DLL path.
