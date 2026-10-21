# windows_arm64

Drop the stripped `libwelvet` (or `.dll` / `.dylib`) for this platform here after cross/native build from welvet.cpp `0.1.85`.

Expected names:
- linux_*: `libwelvet.so`
- darwin_*: `libwelvet.dylib`
- windows_*: `welvet.dll` (+ `libwelvet.a` import lib if needed)

Then bump `native/cpp/MANIFEST.txt` and wire `nativecpp/*_windows_arm64.go`.
