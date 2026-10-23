# Welvet version map

| Component | Tag now | Role |
|-----------|---------|------|
| **welvet** (Go) | **v1.2.0** | Engine + `nativecpp` / `nativeqnn` bindings · scorecard **100/100** |
| **welvet.cpp** | **0.67** | CPU + C ABI — **same 100-pt board as Go**; `0.{round(earned)}` until 1.0 |
| **welvet.qnn** | **1.0.0** | Qualcomm Hexagon HTP (sibling; not on Go/cpp engine boards) |

## welvet.cpp vs Go

cpp uses Go’s ten categories. Today **67/100** → tag **0.67** (layers + FormatNone×34 + train deep; Model/IO + most Systems still open). That is why it is much higher than 0.2, and why it is not 1.0 yet.

Future vendor tags: see [README § Vendor versioning](README.md#vendor-versioning--future).
