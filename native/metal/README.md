# Independent Apple Metal backend

Go Welvet **v1.3.0** introduced this optional binding; **v1.3.1** pins the
bundled engine to **welvet.metal 0.70** (View/Flatten + Mesh* sparse-host train)
with model ABI **1**. Tagging/publication is separate from the working tree.

`darwin_arm64/libwelvet_metal.dylib` is built from welvet.metal, not welvet.cpp.
The public session ABI is in `include/welvet_metal.h`; version/checksum metadata
is in `MANIFEST.txt`. The current artifact targets macOS 13 and was verified on
Apple M5. Other Apple hardware/OS releases need their own execution checks.

Go usage:

```go
session, err := nativemetal.Mount(model)
// Handle err, then defer session.Close().
loss, err := session.TrainMSE(input, target, parallel.ModeNormalBP, learningRate)
// Go parameters are unchanged until:
err = session.SyncToGo()
// Now save through Go's existing ENTITY/model writer.
```

Build Go applications with `-tags welvet_metal` to disable the CPU sibling's cgo
binding. The Metal binding uses dlopen and a versioned C ABI; no CPU training
fallback occurs. Put the dylib beside a distributed executable, or select it
with `nativemetal.Options.LibraryPath` / `WELVET_METAL_LIBRARY`. The installed
checkout path is also used during development.

Rebuild/install from the sibling Metal checkout:

```sh
cd ../welvet.metal
./scripts/macos_install_go.sh
./scripts/macos_build_w2a.sh --go-integration
```

The complete runnable Go example is `cmd/welvet-metal-example`:

```sh
go run -tags welvet_metal ./cmd/welvet-metal-example -save trained.entity
```

The supported boundary is Float32 tensors and FormatNone native parameters
across 34 dtypes (GDN Float32). Layer/Stack/Parallel/Grid mounting, explicit
parameter return, persistent Stack Step/proxy state, shared-volume rules and
seven mesh-clock credits are supported. Transient native state is not exported
as a checkpoint. Calls use session-owned copies rather than Go/GPU zero-copy.

Packed quant stores, tied mutable parameters, heterogeneous Sequential/Residual
(use Stack), advanced unencoded MHA/Softmax options, custom/live Meta rules,
live GDN recurrent state and automatic hardware/performance selection remain
outside this mounted descriptor. Unsupported mounts return errors. See the
Metal repository's `docs/GO_INTEGRATION.md` for ownership, lifecycle and precise
limits. Close never implicitly saves or synchronizes Go weights.

Automatic execution of explicit JSON/`Exec.Backend` assignments is available via
`nativemetal.EnableRouting()`; close the returned router after the workload.
Use `runtime/placement.Stack` or `.Grid` to apply the existing execution JSON.
Routed native updates synchronize to Go automatically; explicit `Mount` sessions
still require `SyncToGo`. The normal build supports Go/C++/Metal simultaneously;
`-tags welvet_metal` disables C++. Example: `go run ./cmd/welvet-metal-routing-example`.
The source checkout's `docs/GO_INTEGRATION.md` describes ownership and limits.
