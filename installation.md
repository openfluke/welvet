# Welvet installation

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

### Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `undefined: Texture`, `Surface`, or related WebGPU types | In this setup, cgo was disabled, excluding the C-backed definitions. Check `go env CGO_ENABLED`; enable cgo and configure the compiler. |
| `C compiler "gcc" not found` | cgo is enabled, but Go cannot find its default compiler. Install llvm-mingw and save `CC`/`CXX` as above. GCC specifically is not required. |
| `exit status 0xc0000135` when starting the built program | A runtime DLL could not be found. This setup needed Welvet's native DLL directory and llvm-mingw's `bin` directory on PATH. |
| New settings do not appear in a terminal | Restart the parent terminal application; check for explicit environment overrides. |

No debug flag was needed.
