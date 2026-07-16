// Package lstm is Welvet LSTM (loom / PyTorch tanh+sigmoid gates).
//
// Layout: input/output [batch, seq, dim]; four gates (i,f,g,o) each with
// W_ih / W_hh via Dense (+ bias on IH). FormatNone×34 + all quants × backends.
// pre stores [i,f,g,o,c] × hidden (5·H) per timestep for BPTT.
//
// Contract: CPU tiled + SIMD + WebGPU, native dtype × k-quant forward/backward.
// No QAT. Tests/docs/CABI live in github.com/openfluke/w2a — not here.
package lstm
