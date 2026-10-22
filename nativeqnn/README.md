# nativeqnn

Go FFI for **welvet.qnn** (`libwelvet_qnn`). Build tag: `cgo && windows && arm64`.

```go
import (
    "github.com/openfluke/welvet/core"
    "github.com/openfluke/welvet/layers/dense"
    "github.com/openfluke/welvet/nativeqnn"
)

l, _ := dense.New(4, 8, core.ActivationLinear, core.DTypeFloat32)
if err := l.SyncToQNN(); err != nil { /* caps reject */ }
l.Exec.Backend = core.BackendQNN
// Forward[float32](l, x) → Hexagon HTP
_ = l.SyncFromQNN() // optional pull-back
_ = nativeqnn.SupportedLayers()
```

| API | Role |
|-----|------|
| `Available` / `Version` / `HasSDK` | DLL / build |
| `LayerSupported` / `SupportedLayers` | HTP caps |
| `OpenHTP` / `DefaultSession` | NPU session |
| `WeightMirror` / layer `SyncToQNN` | user weight push |
| `Sequential` | fused stack push/forward |
