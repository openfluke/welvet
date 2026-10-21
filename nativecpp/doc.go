// Package nativecpp binds welvet.cpp via the C ABI for optional native acceleration.
//
// On linux/amd64 with cgo, all WELVET_API symbols in welvet_c.h are wrapped
// (forward + NormalBP for signed kinds, Parallel forward, Stack/Parallel/Grid/Line TrainMSE by TrainMode, Stack StepBP, version/gaps).
// Other platforms build stubs that return an unavailable error.
//
// Configuration (JSON or Go) resolves Network → Volume → Sector → Layer backends.
// Resolve() empty default is go_cpu; core.DefaultBackend() is BackendCPUTiled.
// PreferNativeCPU() / Exec.Backend=BackendNativeCPU opt into the C ABI when libwelvet is loaded.
package nativecpp
