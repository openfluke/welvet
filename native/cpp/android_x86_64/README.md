# android_x86_64

Shipped `libwelvet.so` for Android x86_64 (emulator / API 28+). Artifacts only — NativeCPU cgo not wired yet.

```bash
cd welvet.cpp
export ANDROID_NDK=/path/to/ndk
cmake -S . -B build-android-x86_64 \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=x86_64 -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release -DWELVET_BUILD_TOOLS=OFF -DWELVET_BUILD_EXAMPLES=OFF \
  -DWELVET_ENABLE_AVX2=OFF
cmake --build build-android-x86_64 -j
llvm-strip --strip-unneeded -o ../welvet/native/cpp/android_x86_64/libwelvet.so \
  build-android-x86_64/libwelvet.so
```

Engine pin: see `../MANIFEST.txt`.
