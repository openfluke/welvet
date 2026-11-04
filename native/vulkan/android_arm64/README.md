# android_arm64

Shipped `libwelvet_vulkan.so` for Android (arm64-v8a, API 28+). Artifacts for
`jniLibs` / NDK packaging. Go `nativevulkan` can dlopen this path on
`GOOS=android` when cgo is enabled.

```bash
cd welvet.vulkan
./scripts/android_install_go.sh
```

Engine pin: see `../MANIFEST.txt`.
