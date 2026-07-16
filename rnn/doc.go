// Package rnn is Welvet vanilla RNN (loom / PyTorch tanh cell).
//
// Layout: input/output [batch, seq, dim]; weights W_ih / W_hh via Dense
// projections (FormatNone×34 + all quants × CPU/SIMD/WebGPU) + bias on IH.
// Forward: h_t = tanh(W_ih x_t + W_hh h_{t-1} + b), h_0 = 0.
//
// Contract: CPU tiled + SIMD + WebGPU, native dtype × k-quant forward/backward.
// No QAT. Tests/docs/CABI live in github.com/openfluke/w2a — not here.
package rnn
