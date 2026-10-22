# android_arm64

Shipped `libwelvet.so` for Android arm64-v8a (API 28+). Artifacts only — NativeCPU cgo not wired yet; drop into `jniLibs/arm64-v8a` or link from an Android NDK host.

```bash
cd welvet.cpp
export ANDROID_NDK=/path/to/ndk
cmake -S . -B build-android-arm64 \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release -DWELVET_BUILD_TOOLS=OFF -DWELVET_BUILD_EXAMPLES=OFF
cmake --build build-android-arm64 -j
llvm-strip --strip-unneeded -o ../welvet/native/cpp/android_arm64/libwelvet.so \
  build-android-arm64/libwelvet.so
```

Engine pin: see `../MANIFEST.txt`.
