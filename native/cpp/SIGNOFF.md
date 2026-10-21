# Platform sign-off (go_v_cpp `--all-train`)

Engine pin: **0.1.85-m2-grid-line-trainmodes** (`native/cpp/MANIFEST.txt`).
Go Welvet: **v1.1.8** (`nativecpp` + `linux_amd64/libwelvet.so` + `windows_arm64/libwelvet.dll`).

| Platform | Status | Sections | Total wall | Date |
|----------|--------|----------|------------|------|
| **linux/amd64** | **PASS** (suite vs 0.1.82; re-run optional for 0.1.85) | **52 / 52** | **51.1s** | 2026-10-21 |
| linux/arm64 | pending | | | |
| darwin/arm64 | pending | | | |
| windows/amd64 | pending | | | |
| windows/arm64 | **lib+tool shipped**; AllTrain **5/52** PASS (Dense/RMS/LN/SwiGLU/LSTM); rest mostly 1-ulp exact_bit on ARM | see logs | **6m59s** | 2026-10-22 |

## How to re-run on another machine

```bash
# 1) Build welvet.cpp on that OS/arch
cd welvet.cpp && ./scripts/build.sh

# 2) Install one stripped lib into Go welvet (adjust name/ext for .dylib/.dll)
strip -o ../welvet/native/cpp/<goos>_<goarch>/libwelvet.so build/libwelvet.so
# Ship a single unversioned libwelvet.so (SONAME is libwelvet.so).

# 3) Sign-off suite (timing table + section totals)
cd welvet.cpp && ./scripts/linux_build_w2a.sh --all-train
# or: WELVET_DENSE_FWD=$PWD/build/welvet_dense_fwd go test ./tests/go_v_cpp/ -run '^TestGoVCppAllTrain$' -v -count=1 -timeout 120m
```

Update this table when a platform passes.
