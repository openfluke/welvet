# android_arm64 — welvet.qnn (Hexagon NPU)

Shipped `libwelvet_qnn.so` for Android arm64-v8a (API 28+). Artifacts for
`jniLibs` / NDK packaging. Go `nativeqnn` links this path on
`GOOS=android` + `arm64` when cgo is enabled.

Built with **QAIRT** headers (HTP graphs enabled). Runtime still needs QAIRT
Android HTP libs + Hexagon skels packaged with the app (or on device).

```bash
cd welvet.qnn
export QAIRT_SDK_ROOT=/path/to/qairt/...
./scripts/android_install_go.sh
```

Engine pin: see `../MANIFEST.txt` (welvet.qnn **1.1.0**).
