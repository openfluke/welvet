// Package sequential is Welvet Sequential compose (loom nested child chain).
//
// One cell Op that runs an ordered list of Dense children: y = f_n(…f_1(x)…).
// Does not duplicate grid hop order — nests inside one BindOp.
// Weights are the concat of child Dense stores (FormatNone×34 + all quants).
//
// Contract: CPU tiled + SIMD + WebGPU via child Dense. No QAT. Tests in w2a.
package sequential
