# ios_arm64

Shipped static `libwelvet_metal.a` for iPhoneOS arm64 (deployment 15.0+).
Link into the app binary with `-force_load` plus `-framework Metal -framework Foundation`;
no shared dylib on device.

```bash
cd welvet.metal
./scripts/ios_install_go.sh
```

Go `nativemetal` dlopen path is macOS-only today; this artifact is for Xcode / gomobile app link.
