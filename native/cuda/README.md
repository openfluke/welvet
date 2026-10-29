# Independent NVIDIA CUDA backend

Go Welvet **v1.4.0** introduces this optional binding and pins the bundled
engine to **welvet.cuda 0.79** with model ABI **1**. Tagging/publication is
separate from the working tree.

`linux_x86_64/libwelvet_cuda.so` and `windows_amd64/libwelvet_cuda.dll` are
built from welvet.cuda, not welvet.cpp. The public session ABI is in
`include/welvet_cuda.h`; version/checksum metadata is in `MANIFEST.txt` and
per-platform `SHA256SUMS`. Verified hosts: Linux x86_64 and Windows amd64 with
an NVIDIA GPU + matching driver/runtime. Other CUDA platforms need their own
execution checks.

Go usage:

```go
session, err := nativecuda.Mount(model)
// Handle err, then defer session.Close().
loss, err := session.TrainMSE(input, target, parallel.ModeNormalBP, learningRate)
// Go parameters are unchanged until:
err = session.SyncToGo()
// Now save through Go's existing ENTITY/model writer.
```

Build Go applications with `-tags welvet_cuda` to disable the CPU sibling's cgo
binding. The CUDA binding uses dlopen/LoadLibrary and a versioned C ABI; no CPU
training fallback occurs. Put the shared library beside a distributed
executable, or select it with `nativecuda.Options.LibraryPath` /
`WELVET_CUDA_LIBRARY`. The installed checkout path is also used during
development. On Windows, ensure the CUDA runtime (`cudart`) resolves via `PATH`.

Rebuild/install from the sibling CUDA checkout:

```sh
cd ../welvet.cuda
./scripts/linux_install_go.sh          # Linux x86_64
# ./scripts/windows_install_go.ps1     # Windows amd64 (MSVC + CUDA Toolkit)
```

Supported boundary matches the Metal-style mounted descriptor: FormatNone
native parameters across the dtype matrix Welvet.cuda validates, layer / Stack /
Parallel / Grid mounting, explicit parameter return, and train-mode coverage
through the CUDA scoreboard (**0.79** = StepTweenSplitSparse). Transient native
state is not exported as a checkpoint. Unsupported mounts return errors. See
the CUDA repository's `docs/GO_INTEGRATION.md` for ownership, lifecycle and
precise limits. Close never implicitly saves or synchronizes Go weights.

Automatic execution of explicit JSON/`Exec.Backend` assignments is available via
`nativecuda.EnableRouting()`; close the returned router after the workload.
Use `runtime/placement.Stack` or `.Grid` to apply the existing execution JSON.
Routed native updates synchronize to Go automatically; explicit `Mount` sessions
retain the explicit `SyncToGo` contract.
