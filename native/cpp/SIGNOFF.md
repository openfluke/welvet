# Platform sign-off (go_v_cpp `--all-train`)

Engine pin: **0.1.82-m2-parallel-cascade** (`native/cpp/MANIFEST.txt`).
Go Welvet: **v1.1.4** (`nativecpp` + linux_amd64 `libwelvet.so`).

| Platform | Status | Sections | Total wall | Date |
|----------|--------|----------|------------|------|
| **linux/amd64** | **PASS** | **52 / 52** | **51.1s** | 2026-10-21 |
| linux/arm64 | pending | | | |
| darwin/arm64 | pending | | | |
| windows/amd64 | pending | | | |
| windows/arm64 | pending | | | |

## How to re-run on another machine

```bash
# 1) Build welvet.cpp on that OS/arch
cd welvet.cpp && ./scripts/build.sh

# 2) Install stripped lib into Go welvet (adjust ext for .dylib/.dll)
strip -o ../welvet/native/cpp/<goos>_<goarch>/libwelvet.so build/libwelvet.so.*
# linux: also `ln -sfn libwelvet.so libwelvet.so.0` in that dir (SONAME)

# 3) Sign-off suite (timing table + section totals)
cd welvet.cpp && ./scripts/linux_build_w2a.sh --all-train
# or: WELVET_DENSE_FWD=$PWD/build/welvet_dense_fwd go test ./tests/go_v_cpp/ -run '^TestGoVCppAllTrain$' -v -count=1 -timeout 120m
```

Update this table when a platform passes.
