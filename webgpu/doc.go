// Package webgpu hosts Welvet GPU context and dense GEMV dispatch.
//
// Real wgpu device only. No host “parity” fallback — missing adapter or
// unbound kernels return errors (fix the bind, or user hardware limit).
package webgpu
