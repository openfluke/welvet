# Welvet version map

| Component | Tag now | Role |
|-----------|---------|------|
| **welvet** (Go) | **v1.3.1** | Engine + `nativecpp` / `nativeqnn` / `nativemetal` bindings · scorecard **100/100** |
| **welvet.cpp** | **0.67** | CPU + C ABI — **same 100-pt board as Go**; `0.{round(earned)}` until 1.0 |
| **welvet.metal** | **0.70**, ABI **1** | Independent Apple Metal backend; View/Flatten + Mesh* sparse-host train |
| **welvet.qnn** | **1.0.0** | Qualcomm Hexagon HTP (sibling; not on Go/cpp engine boards) |

## welvet.cpp vs Go

cpp uses Go’s ten categories. Today **67/100** → tag **0.67** (layers + FormatNone×34 + train deep; Model/IO + most Systems still open). That is why it is much higher than 0.2, and why it is not 1.0 yet.

Future vendor tags: see [README § Vendor versioning](README.md#vendor-versioning--future).

## Go v1.2.0 → v1.3.0

After the engine reached 100/100, additive capabilities use minor releases;
compatible fixes/artifact refreshes use patches. The optional Metal binding is
a new capability, so **v1.3.0** follows **v1.2.0**, with no major bump or board
reset. v1.2.0's Qualcomm integration remains available.

v1.3.0 adds `nativemetal.Mount`, repeated native forward/training,
explicit `SyncToGo`, native parameter/bias preservation and Go ENTITY
save/reload. The bundled Metal engine was independently versioned at
**0.68** and the mounted-model ABI at **1**. Existing Linux/Windows and CPU
backends keep their existing platform behavior. Metal-only applications select
`-tags welvet_metal`; merely installing Go Welvet does not select Metal.

## Go v1.3.0 → v1.3.1

**v1.3.1** is a patch: pin the bundled Metal library to **welvet.metal 0.70**
(View/Flatten wire 24/25; Mesh* TrainMSE sparse-host volume fallback) and encode
`parallel.View` / `Flatten` in `nativemetal` so creature MidDepth stacks Mount.
Engine board stays **100/100**; mounted-model ABI stays **1**.

This is the release target prepared in the working tree. Updating these files
does not create a Git tag, push commits, publish a GitHub release or make
`go get ...@v1.3.1` available. `release.sh` reads the README version when a
separate publication is requested.
