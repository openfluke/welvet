# welvet.cpp native artifacts (shipped with Go Welvet)

Prebuilt C ABI libraries for **optional** native acceleration. **Default remains Go `BackendCPUTiled`.** Pure Go runs when the platform lib is missing.

Set `layer.Exec.Backend = core.BackendNativeCPU` (or JSON `"backend": "native_cpu"`, or `core.PreferNativeCPU()` / `dense.ApplyExecConfig`) to route FormatNone f32 Forward / NormalBP through `libwelvet`.

Signed kinds dispatched via C ABI when `BackendNativeCPU` is set (linux/amd64 + cgo):

Dense · MHA · SwiGLU · RMSNorm · LayerNorm · CNN1/2/3 · ConvT1/2/3 · RNN · LSTM · Embedding · Softmax · KMeans · Sequential · Residual · Mamba · Metacognition · GDN · Stack NormalBP · Parallel forward · Stack StepBP dense · version/gaps

Composite layers without a packed C call still inherit NativeCPU onto Dense children (exact_bit MatVec).

| Path | Artifact |
|------|----------|
| `include/welvet_c.h` | Stable C ABI header |
| `linux_amd64/libwelvet.so` | Linux x86-64 shared lib (+ `libwelvet.so.0` SONAME link) |
| `linux_arm64/` | pending |
| `darwin_arm64/` | pending |
| `windows_amd64/` | pending |
| `windows_arm64/` | pending |

Pinned engine: see `MANIFEST.txt` (must match `welvet_version()`).

Go binding: package `github.com/openfluke/welvet/nativecpp`.
