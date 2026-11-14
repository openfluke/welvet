# android_arm64 — welvet.qnn (Hexagon NPU)

Shipped `libwelvet_qnn.so` for Android arm64-v8a (API 28+). Artifacts for
`jniLibs` / NDK packaging. Go `nativeqnn` links this path on
`GOOS=android` + `arm64` when cgo is enabled.

```bash
cd welvet.qnn
./scripts/android_install_go.sh
```

Runtime needs QAIRT Android HTP libs + Hexagon skels packaged with the app
(or on device). Rebuild with `QAIRT_SDK_ROOT` set for a full HTP binary.

Engine pin: see `../MANIFEST.txt` (welvet.qnn **1.1.0**).
