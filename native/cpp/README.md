# welvet.cpp native artifacts (shipped with Go Welvet)

Prebuilt C ABI libraries for **optional** native acceleration. **Default remains Go `BackendCPUTiled`.** Pure Go runs when the platform lib is missing.

Set `layer.Exec.Backend = core.BackendNativeCPU` (or JSON `"backend": "native_cpu"`, or `core.PreferNativeCPU()` / `dense.ApplyExecConfig`) to route FormatNone f32 Forward / NormalBP / TrainMSE through `libwelvet`.

Signed kinds dispatched via C ABI when `BackendNativeCPU` is set (linux/amd64 + cgo):

Dense · MHA · SwiGLU · RMSNorm · LayerNorm · CNN1/2/3 · ConvT1/2/3 · RNN · LSTM · Embedding · Softmax · KMeans · Sequential · Residual · Mamba · Metacognition · GDN · Stack NormalBP · Parallel forward/NormalBP · **Stack/Parallel TrainMSE by TrainMode** · **Grid Mesh TrainMSE** · **Line TrainMSE by TrainMode** · Stack StepBP dense · version/gaps

`TrainStackMSE` / `TrainMSE` intercept when `BackendNativeCPU`: C ABI path + Mirror Capture; call `SyncWeightsFromNative` before `.entity` write. Defaults stay `BackendCPUTiled`.

Composite layers without a packed C call still inherit NativeCPU onto Dense children (exact_bit MatVec).

| Path | Artifact |
|------|----------|
| `include/welvet_c.h` | Stable C ABI header |
| `linux_amd64/libwelvet.so` | Linux x86-64 shared lib (single unversioned artifact) |
| `linux_arm64/libwelvet.so` | Linux aarch64 shared lib |
| `darwin_arm64/libwelvet.dylib` | macOS Apple Silicon shared lib |
| `android_arm64/libwelvet.so` | Android arm64-v8a shared lib (artifact; cgo pending) |
| `android_x86_64/libwelvet.so` | Android x86_64 shared lib (artifact; cgo pending) |
| `ios_arm64/libwelvet.a` | iPhoneOS arm64 static archive (artifact; cgo pending) |
| `windows_amd64/` | `libwelvet.dll` (+ import lib) |
| `windows_arm64/` | `libwelvet.dll` (+ import lib) |

Pinned engine: see `MANIFEST.txt` (must match `welvet_version()`). Current: **0.67**.

Go binding: package `github.com/openfluke/welvet/nativecpp`.
