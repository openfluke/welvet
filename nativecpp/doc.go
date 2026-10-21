// Package nativecpp binds welvet.cpp via the C ABI for optional native acceleration.
//
// On linux/amd64 with cgo, all WELVET_API symbols in welvet_c.h are wrapped
// (forward + NormalBP for signed kinds, Parallel forward, Stack StepBP, version/gaps).
// Other platforms build stubs that return an unavailable error.
//
// Configuration (JSON or Go) resolves Network → Volume → Sector → Layer backends.
// When libwelvet is loaded, Resolve() defaults to native_cpu and core.PreferNativeCPU()
// selects BackendNativeCPU so layers dispatch through the C ABI without manual config.
// Missing platform libs fall back to pure Go (go_cpu / BackendCPUTiled).
package nativecpp
