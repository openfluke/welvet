// Package nativecpp binds welvet.cpp via the C ABI for optional native acceleration.
//
// Default layer Exec.Backend remains BackendCPUTiled (pure Go). Set BackendNativeCPU
// to route FormatNone f32 work through libwelvet. After native train, call
// SyncWeightsFromNative() to pull W_after into Go stores for .entity save.
//
// JSON/Go ExecConfig resolves Network→Volume→Sector→Layer; empty default is go_cpu.
package nativecpp
