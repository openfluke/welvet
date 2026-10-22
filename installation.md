# Welvet installation

How to install Go Welvet and (optionally) enable the shipped **NativeCPU** C ABI (`libwelvet`) via cgo.

| Platform | Native artifact | Go `nativecpp` cgo | Notes |
| --- | --- | --- | --- |
| **windows/arm64** | `libwelvet.dll` | **wired** | Procedure below **verified** on device |
| **windows/amd64** | `libwelvet.dll` | **wired** | Same pattern as ARM64; llvm-mingw x86_64 |
| **linux/amd64** | `libwelvet.so` | **wired** | Usual desktop / CI path |
| **linux/arm64** | `libwelvet.so` | **wired** | Fedora / WSL / native aarch64 |
| **darwin/arm64** | `libwelvet.dylib` | **wired** | Apple Silicon |
| **android/arm64** | `libwelvet.so` | artifact only | Drop into `jniLibs`; cgo pending |
| **android/x86_64** | `libwelvet.so` | artifact only | Emulator; cgo pending |
| **ios/arm64** | `libwelvet.a` | artifact only | Static link / `-force_load`; cgo pending |

**Defaults:** pure Go `BackendCPUTiled` always works without a C compiler. Native acceleration needs cgo + the platform lib (or packaging that lib next to your binary).

Pin the module (any OS):

```bash
go get github.com/openfluke/welvet@v1.1.10
```

Engine pin for shipped natives: **0.1.85-m2-grid-line-trainmodes** (`native/cpp/MANIFEST.txt`).

---

## Windows ARM64

This procedure was verified on Windows ARM64 using:

| Component | Version |
| --- | --- |
| Go | 1.27.3, windows/arm64 |
| Welvet | v1.1.10 |
| WebGPU dependency | github.com/openfluke/webgpu v1.0.6 |
| C/C++ toolchain | llvm-mingw 20261020, UCRT, aarch64 host |

The verified example trains a small model on the CPU, saves it as an ENTITY file, reloads it, and checks predictions. GPU execution was not tested.

### 1. Install Go for Windows ARM64

Download the Windows ARM64 archive from the [official Go downloads](https://go.dev/dl/). This setup used `go1.27.3.windows-arm64.zip`.

Compare the archive's SHA-256 with the official download metadata before extracting:

```powershell
Get-FileHash .\go1.27.3.windows-arm64.zip -Algorithm SHA256
```

Extract the archive into `C:\git\toolchains`, producing:

```text
C:\git\toolchains\go\bin\go.exe
```

Add `C:\git\toolchains\go\bin` to your Windows **user PATH**. In PowerShell:

```powershell
function Add-UserPath([string]$Directory) {
    $savedPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (($savedPath -split ';') -notcontains $Directory) {
        $updatedPath = ($savedPath.TrimEnd(';') + ';' + $Directory).TrimStart(';')
        [Environment]::SetEnvironmentVariable('Path', $updatedPath, 'User')
    }
}

Add-UserPath 'C:\git\toolchains\go\bin'
$env:Path = 'C:\git\toolchains\go\bin;' + $env:Path
go version
```

Expected architecture: `windows/arm64`.

### 2. Install the ARM64 C/C++ compiler

Download `llvm-mingw-20261020-ucrt-aarch64.zip` from the [official llvm-mingw release](https://github.com/mstorsjo/llvm-mingw/releases/tag/20261020). Verify its SHA-256 against the release asset metadata, then extract it into `C:\git\toolchains`.

The compiler used here is:

```text
C:\git\toolchains\llvm-mingw-20261020-ucrt-aarch64\bin\aarch64-w64-mingw32-clang.exe
```

Welvet includes native binaries, but Go still needs a C compiler to compile its cgo bridge and link those libraries. WebGPU v1.0.6 includes the Windows ARM64 gnullvm library; this setup did not rebuild WebGPU.

### 3. Enable cgo and persist the compiler settings

Run the following in the same PowerShell session as step 1:

```powershell
$compilerBin = 'C:\git\toolchains\llvm-mingw-20261020-ucrt-aarch64\bin'
$cc = Join-Path $compilerBin 'aarch64-w64-mingw32-clang.exe'
$cxx = Join-Path $compilerBin 'aarch64-w64-mingw32-clang++.exe'

& $cc --version

# Go user settings: used by Go in CMD and PowerShell.
go env -w CGO_ENABLED=1 "CC=$cc" "CXX=$cxx"

# Windows user environment: inherited by new terminal processes.
[Environment]::SetEnvironmentVariable('CGO_ENABLED', '1', 'User')
[Environment]::SetEnvironmentVariable('CC', $cc, 'User')
[Environment]::SetEnvironmentVariable('CXX', $cxx, 'User')
Add-UserPath $compilerBin

# Also update this running PowerShell session.
$env:CGO_ENABLED = '1'
$env:CC = $cc
$env:CXX = $cxx
$env:Path = $compilerBin + ';' + $env:Path
```

These are user-wide defaults. An explicit environment override in a terminal or script takes precedence over Go's saved settings.

### 4. Create a separate Welvet project

For a new project, initialize a module and pin Welvet:

```powershell
New-Item -ItemType Directory -Force C:\git\welvet-1.1.10-demo
Set-Location C:\git\welvet-1.1.10-demo
go mod init example.com/welvet-arm64-demo
go get github.com/openfluke/welvet@v1.1.10
```

Skip initialization if the project already has a `go.mod`. After adding your Go source, run `go mod tidy`. In this demo, Welvet v1.1.10 resolves WebGPU v1.0.6.

### 5. Make the bundled Welvet DLL discoverable

The executable imports `libwelvet.dll`. Add the directory containing the bundled Windows ARM64 DLL to the user PATH:

```powershell
$moduleCache = go env GOMODCACHE
$nativeBin = Join-Path $moduleCache 'github.com\openfluke\welvet@v1.1.10\native\cpp\windows_arm64'
if (!(Test-Path (Join-Path $nativeBin 'libwelvet.dll'))) {
    throw "Welvet ARM64 DLL not found in $nativeBin"
}
Add-UserPath $nativeBin
$env:Path = $nativeBin + ';' + $env:Path
```

Keep the llvm-mingw `bin` directory on PATH as well: the built executable also imports its `libunwind.dll`.

This PATH entry points into the Go module cache for **v1.1.10**. If you change Welvet versions or clear that cache, update or restore the DLL directory. For a distributed application, package its required runtime DLLs with the executable instead of relying on this development-machine PATH.

### 6. Restart terminals and verify

Fully close and reopen your terminal application, including Windows Terminal or VS Code if it hosts your terminals. Existing processes retain their earlier environment.

The following commands work in both CMD and PowerShell:

```text
go version
go env CGO_ENABLED CC CXX
cd C:\git\welvet-1.1.10-demo
go run .
```

`CGO_ENABLED` should report `1`, and `CC`/`CXX` should point to the ARM64 llvm-mingw compilers.

The included `main.go` uses CPU tiled execution and learns `y = 2x + 1`. The verified run produced:

```text
epoch 1 MSE: 3.000000000
epoch 500 MSE: 0.000000000
x=-0.75 predicted=-0.499999 expected=-0.500000
x= 0.25 predicted=1.500000 expected=1.500000
x= 2.00 predicted=4.999997 expected=5.000000
PASS: trained y=2x+1, saved linear.entity, reloaded and verified unseen inputs.
```

The saved model is `linear.entity`. The complete successful output is in `latest-run.txt`.

### Troubleshooting (Windows ARM64)

| Symptom | Cause and fix |
| --- | --- |
| `undefined: Texture`, `Surface`, or related WebGPU types | In this setup, cgo was disabled, excluding the C-backed definitions. Check `go env CGO_ENABLED`; enable cgo and configure the compiler. |
| `C compiler "gcc" not found` | cgo is enabled, but Go cannot find its default compiler. Install llvm-mingw and save `CC`/`CXX` as above. GCC specifically is not required. |
| `exit status 0xc0000135` when starting the built program | A runtime DLL could not be found. This setup needed Welvet's native DLL directory and llvm-mingw's `bin` directory on PATH. |
| New settings do not appear in a terminal | Restart the parent terminal application; check for explicit environment overrides. |

No debug flag was needed.

---

## Windows AMD64 (x86-64)

Same shape as Windows ARM64: Go + **llvm-mingw** (x86_64 host) + PATH to `native/cpp/windows_amd64` so `libwelvet.dll` loads.

### 1. Go

Install the Windows AMD64 toolchain from [go.dev/dl](https://go.dev/dl/) (e.g. `go1.22+` / current stable `windows-amd64.zip`). Put `go\bin` on user PATH. Confirm:

```powershell
go version
# expect: windows/amd64
```

### 2. C compiler (llvm-mingw x86_64)

Download the **ucrt-x86_64** llvm-mingw zip from [mstorsjo/llvm-mingw releases](https://github.com/mstorsjo/llvm-mingw/releases) (same release tag family as ARM64 is fine). Extract under e.g. `C:\git\toolchains`.

Typical compilers:

```text
...\llvm-mingw-*-ucrt-x86_64\bin\x86_64-w64-mingw32-clang.exe
...\llvm-mingw-*-ucrt-x86_64\bin\x86_64-w64-mingw32-clang++.exe
```

### 3. Enable cgo

```powershell
$compilerBin = 'C:\git\toolchains\llvm-mingw-<ver>-ucrt-x86_64\bin'  # adjust
$cc = Join-Path $compilerBin 'x86_64-w64-mingw32-clang.exe'
$cxx = Join-Path $compilerBin 'x86_64-w64-mingw32-clang++.exe'

go env -w CGO_ENABLED=1 "CC=$cc" "CXX=$cxx"
[Environment]::SetEnvironmentVariable('CGO_ENABLED', '1', 'User')
[Environment]::SetEnvironmentVariable('CC', $cc, 'User')
[Environment]::SetEnvironmentVariable('CXX', $cxx, 'User')
# Also add $compilerBin to user PATH (same Add-UserPath helper as ARM64)
```

### 4. Project + DLL PATH

```powershell
go mod init example.com/welvet-amd64-demo
go get github.com/openfluke/welvet@v1.1.10

$moduleCache = go env GOMODCACHE
$nativeBin = Join-Path $moduleCache 'github.com\openfluke\welvet@v1.1.10\native\cpp\windows_amd64'
# Add $nativeBin (and llvm-mingw bin) to PATH — same pattern as ARM64 step 5
```

Ship `libwelvet.dll` (and mingw runtime DLLs if needed) next to your `.exe` for distribution.

Sign-off helper in the suite: `w2a/scripts/run_go_v_cpp_windows_snapshot.cmd` → `logs/windowsx86.txt`.

---

## Linux AMD64

### 1. Go + C toolchain

```bash
# Distro packages (Fedora example)
sudo dnf install golang gcc gcc-c++   # or go from go.dev

go version   # expect linux/amd64
```

cgo needs a working `gcc`/`clang` on `PATH`. Default `CGO_ENABLED=1` on Linux.

### 2. Project

```bash
mkdir -p ~/welvet-demo && cd ~/welvet-demo
go mod init example.com/welvet-linux-amd64-demo
go get github.com/openfluke/welvet@v1.1.10
go mod tidy
```

### 3. Native lib

cgo links `native/cpp/linux_amd64` with an **rpath** into the module cache, so you usually do **not** need `LD_LIBRARY_PATH` for local `go run` / `go test`.

Optional check:

```bash
ls "$(go env GOMODCACHE)/github.com/openfluke/welvet@v1.1.10/native/cpp/linux_amd64/libwelvet.so"
```

For a deployed binary, either keep rpath valid or install/`patchelf`/`LD_LIBRARY_PATH` so `libwelvet.so` resolves.

### 4. Verify

```bash
go env CGO_ENABLED
go run .
```

Optional NativeCPU: set `layer.Exec.Backend = core.BackendNativeCPU` (or `core.PreferNativeCPU()`). Default stays `BackendCPUTiled`.

---

## Linux ARM64

Same as linux/amd64, different artifact directory.

```bash
# Confirm arch
uname -m    # aarch64
go env GOARCH   # arm64

go get github.com/openfluke/welvet@v1.1.10
ls "$(go env GOMODCACHE)/github.com/openfluke/welvet@v1.1.10/native/cpp/linux_arm64/libwelvet.so"
```

Needs `gcc`/`clang` + `CGO_ENABLED=1`. Rpath points at `linux_arm64`. Rebuild recipe if you refresh the lib from welvet.cpp: see `native/cpp/linux_arm64/README.md`.

---

## macOS Darwin ARM64 (Apple Silicon)

### 1. Go + Xcode CLT

```bash
# Go: https://go.dev/dl/  (darwin-arm64) or brew install go
xcode-select --install   # clang for cgo

go version   # expect darwin/arm64
```

### 2. Project

```bash
mkdir -p ~/welvet-demo && cd ~/welvet-demo
go mod init example.com/welvet-darwin-arm64-demo
go get github.com/openfluke/welvet@v1.1.10
```

### 3. Native lib

cgo links `native/cpp/darwin_arm64/libwelvet.dylib` with rpath. Confirm:

```bash
ls "$(go env GOMODCACHE)/github.com/openfluke/welvet@v1.1.10/native/cpp/darwin_arm64/libwelvet.dylib"
```

### 4. Verify

```bash
CGO_ENABLED=1 go run .
```

Suite snapshot helper (sibling w2a checkout): `w2a/scripts/run_go_v_cpp_mac_snapshot.sh` → `logs/mac.txt`.

---

## Android (arm64-v8a / x86_64)

**Status:** `libwelvet.so` is **shipped** under `native/cpp/android_arm64` and `android_x86_64`. Go **`nativecpp` cgo is not wired yet** — you cannot `PreferNativeCPU` from a normal `GOOS=android` Go build today.

### Use the artifact from an NDK / JNI app

1. `go get github.com/openfluke/welvet@v1.1.10` (or copy from a checkout).
2. Copy the matching `.so` into your app, e.g.:

```text
app/src/main/jniLibs/arm64-v8a/libwelvet.so
app/src/main/jniLibs/x86_64/libwelvet.so
```

3. Load / link via JNI or an NDK shared-lib dependency against `native/cpp/include/welvet_c.h`.

Rebuild from welvet.cpp (API 28+): see `native/cpp/android_arm64/README.md` and `android_x86_64/README.md`.

---

## iOS (arm64)

**Status:** static `libwelvet.a` is **shipped** under `native/cpp/ios_arm64`. Go **`nativecpp` cgo is not wired yet**.

### Link into an app

1. Obtain `libwelvet.a` from the module cache or repo (`native/cpp/ios_arm64/`).
2. Add it to the Xcode / gomobile link line; typically **`-force_load`** the archive (no device dylib).
3. Headers: `native/cpp/include/welvet_c.h`.
4. Deployment target: **15.0+** (as built).

Rebuild from welvet.cpp with iphoneos SDK: see `native/cpp/ios_arm64/README.md`.

---

## Pure Go (any OS, no native)

If you skip cgo or the platform has no wired lib:

```bash
CGO_ENABLED=0 go build ./...
```

Welvet runs on **`BackendCPUTiled`** (and other Go backends). NativeCPU simply stays unavailable (`nativecpp` stub).

---

## Common troubleshooting

| Symptom | Likely fix |
| --- | --- |
| `nativecpp: libwelvet not available on this platform` | Wrong OS/arch, `CGO_ENABLED=0`, or android/ios (cgo pending). |
| `C compiler … not found` | Install gcc/clang (Unix) or llvm-mingw (Windows); set `CC`/`CXX`. |
| Windows `0xc0000135` / missing DLL | Put `native/cpp/windows_*` **and** llvm-mingw `bin` on `PATH`, or copy DLLs next to the exe. |
| Linux/mac: `libwelvet.so` / `.dylib` not found at runtime | Broken rpath after relocating the binary; set `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH` or ship the lib beside the binary. |
| WebGPU types undefined | Often cgo off — enable cgo and a working C toolchain. |

More detail per artifact: `native/cpp/README.md` and each `native/cpp/<platform>/README.md`.
