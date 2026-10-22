# ios_arm64

Shipped static `libwelvet.a` for iPhoneOS arm64 (deployment 15.0+). Link into the app binary (DynamicLibrary.process / `-force_load`); no shared dylib on device.

```bash
cd welvet.cpp
SDK="$(xcrun --sdk iphoneos --show-sdk-path)"
CXX="$(xcrun --sdk iphoneos --find clang++)"
cmake -S . -B build-ios-arm64 -G "Unix Makefiles" \
  -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 -DCMAKE_OSX_SYSROOT="$SDK" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_CXX_FLAGS="-isysroot $SDK -arch arm64 -miphoneos-version-min=15.0" \
  -DCMAKE_BUILD_TYPE=Release -DWELVET_BUILD_SHARED=OFF \
  -DWELVET_BUILD_TOOLS=OFF -DWELVET_BUILD_EXAMPLES=OFF
cmake --build build-ios-arm64 -j
cp -f build-ios-arm64/libwelvet.a ../welvet/native/cpp/ios_arm64/libwelvet.a
```

Engine pin: see `../MANIFEST.txt`. NativeCPU Go cgo not wired for ios yet.
