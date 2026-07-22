// Package weights is the native WeightStore (Format + DType are storage truth).
//
// Morphing between numerical types / quant formats is supported via Convert /
// Converted, which hub through float32 (lossy for low-bit hops). There is no
// silent QAT fake-quant path — Convert is an explicit re-encode.
package weights
