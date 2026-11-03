# Independent Vulkan backend

Go Welvet **v1.6.0** introduces this optional binding and pins the bundled
engine to **welvet.vulkan 0.79** with model ABI **1**. Tagging/publication is
separate from the working tree.

`linux_x86_64/libwelvet_vulkan.so` is built from welvet.vulkan, not welvet.cpp.
The public session ABI is in `include/welvet_vulkan.h`; version/checksum metadata
is in `MANIFEST.txt` and per-platform `SHA256SUMS`. Verified hosts: Linux x86_64
with a Vulkan 1.x-capable GPU + loader/runtime. Windows artifacts are not
shipped yet; other Vulkan platforms need their own execution checks.

Go usage:

```go
session, err := nativevulkan.Mount(model)
// Handle err, then defer session.Close().
loss, err := session.TrainMSE(input, target, parallel.ModeNormalBP, learningRate)
// Go parameters are unchanged until:
err = session.SyncToGo()
// Now save through Go's existing ENTITY/model writer.
```

Build Go applications with `-tags welvet_vulkan` to disable the CPU sibling's cgo
binding. The Vulkan binding uses dlopen and a versioned C ABI; no CPU training
fallback occurs. Put the shared library beside a distributed executable, or
select it with `nativevulkan.Options.LibraryPath` / `WELVET_VULKAN_LIBRARY`.
The installed checkout path is also used during development.

Rebuild/install from the sibling Vulkan checkout:

```sh
cd ../welvet.vulkan
./scripts/linux_install_go.sh          # Linux x86_64
```

Supported boundary matches the Metal-style mounted descriptor: FormatNone
native parameters across the dtype matrix Welvet.cuda validates, layer / Stack /
Parallel / Grid mounting, explicit parameter return, and train-mode coverage
through the Vulkan scoreboard (**0.79** = StepTweenSplitSparse). Transient native
state is not exported as a checkpoint. Unsupported mounts return errors. See
the Vulkan repository's `docs/VULKAN_ONLY.md` and `docs/TRAINING_CHECKLIST.md`
for ownership, purity contract, and train-mode limits. Close never implicitly
saves or synchronizes Go weights.

Automatic execution of explicit JSON/`Exec.Backend` assignments is available via
`nativevulkan.EnableRouting()`; close the returned router after the workload.
Use `runtime/placement.Stack` or `.Grid` to apply the existing execution JSON.
Routed native updates synchronize to Go automatically; explicit `Mount` sessions
retain the explicit `SyncToGo` contract.

**Note:** GPU compute requires a Vulkan-capable device. Build/load is validated;
compute remains environment-dependent until exercised on target hardware.
