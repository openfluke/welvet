package qwentts

// WGSL for the resident Qwen3 talker decode fuse (one compute pass per token).
//
// Adapted from welvet/fusedgpu (RMSNorm, RoPE rotate_half, GQA AttnDecode,
// KVUpdate, Residual, HeadRMS, IncPos) and welvet/apps/mosstts (FP32 GEMV).
// Everything runs FP32 on GPU-resident dense weights loaded from BF16 on host.

// FP32 GEMV with optional bias: Y[o] = sum_i X[i]*W[o*in+i] (+B[o]).
const shaderQwenGEMV = `
struct Params { inputSize: u32, outputSize: u32, hasBias: u32, _pad: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> B: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    if (o >= params.outputSize) { return; }
    var sum: f32 = 0.0;
    let wBase = o * params.inputSize;
    for (var i: u32 = 0u; i < params.inputSize; i++) {
        sum += X[i] * W[wBase + i];
    }
    if (params.hasBias != 0u) {
        sum += B[o];
    }
    Y[o] = sum;
}
`

// RMSNorm (no bias, no mean subtraction): out = x * rsqrt(mean(x^2)+eps) * weight.
const shaderQwenRMSNorm = `
struct Params { dim: u32, eps: f32, _p0: u32, _p1: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let dim = params.dim;
    var local: f32 = 0.0;
    for (var i = tid; i < dim; i += 64u) {
        let v = input[i];
        local += v * v;
    }
    partial[tid] = local;
    workgroupBarrier();
    var stride = 32u;
    while (stride > 0u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inv = inverseSqrt(partial[0] / f32(dim) + params.eps);
    for (var i = tid; i < dim; i += 64u) {
        output[i] = input[i] * inv * weight[i];
    }
}
`

// Per-head RMSNorm (Qwen3 q_norm / k_norm) over headDim, in place on x.
// One workgroup per head; gamma is shared [headDim].
const shaderQwenHeadRMS = `
struct Params { numHeads: u32, headDim: u32, epsBits: u32, _p: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let h = wg_id.x;
    if (h >= params.numHeads) { return; }
    let tid = lid.x;
    let hd = params.headDim;
    let eps = bitcast<f32>(params.epsBits);
    let base = h * hd;
    var local: f32 = 0.0;
    for (var i = tid; i < hd; i += 64u) {
        let v = x[base + i];
        local += v * v;
    }
    partial[tid] = local;
    workgroupBarrier();
    var stride = 32u;
    while (stride > 0u) {
        if (tid < stride) { partial[tid] += partial[tid + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inv = inverseSqrt(partial[0] / f32(hd) + eps);
    for (var i = tid; i < hd; i += 64u) {
        x[base + i] = x[base + i] * inv * gamma[i];
    }
}
`

// Residual: inout_b[i] += a[i].
const shaderQwenResidual = `
struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> inout_b: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    inout_b[i] = inout_b[i] + a[i];
}
`

// RoPE (HF rotate_half): pair index d with d+half; pos read from step[0].
const shaderQwenRoPE = `
struct Params { numHeads: u32, headDim: u32, thetaBits: u32, _pad: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> step: array<u32>; // step[0] = pos
@group(0) @binding(2) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x;
    if (h >= params.numHeads) { return; }
    let pos = step[0];
    let theta = bitcast<f32>(params.thetaBits);
    let half = params.headDim / 2u;
    let base = h * params.headDim;
    for (var i = 0u; i < half; i++) {
        let freq = 1.0 / pow(theta, f32(i * 2u) / f32(params.headDim));
        let ang = f32(pos) * freq;
        let c = cos(ang);
        let s = sin(ang);
        let x0 = x[base + i];
        let x1 = x[base + i + half];
        x[base + i] = x0 * c - x1 * s;
        x[base + i + half] = x0 * s + x1 * c;
    }
}
`

// WriteKV: append k/v for the current pos into caches [kvHeads, maxSeq, headDim].
const shaderQwenWriteKV = `
struct Params { kvDim: u32, maxSeqLen: u32, headDim: u32, _pad: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> step: array<u32>;
@group(0) @binding(2) var<storage, read> kIn: array<f32>;
@group(0) @binding(3) var<storage, read> vIn: array<f32>;
@group(0) @binding(4) var<storage, read_write> kCache: array<f32>;
@group(0) @binding(5) var<storage, read_write> vCache: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.kvDim) { return; }
    let pos = step[0];
    let head = i / params.headDim;
    let d = i % params.headDim;
    let dst = (head * params.maxSeqLen + pos) * params.headDim + d;
    kCache[dst] = kIn[i];
    vCache[dst] = vIn[i];
}
`

// GQA attention decode: one workgroup per query head, softmax over kvLen=pos+1.
const shaderQwenAttn = `
struct Params { numHeads: u32, numKVHeads: u32, headDim: u32, maxSeqLen: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> step: array<u32>;
@group(0) @binding(2) var<storage, read> q: array<f32>;
@group(0) @binding(3) var<storage, read> kCache: array<f32>;
@group(0) @binding(4) var<storage, read> vCache: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

// scores length caps maxSeq at 2048; qv covers headDim<=128.
var<workgroup> qv: array<f32, 128>;
var<workgroup> scores: array<f32, 2048>;
var<workgroup> mbuf: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let h = wg_id.x;
    let tid = lid.x;
    if (h >= params.numHeads) { return; }

    let headDim = params.headDim;
    let kvGroup = params.numHeads / params.numKVHeads;
    let kvH = h / kvGroup;
    let scale = inverseSqrt(f32(headDim));
    let kvLen = step[0] + 1u;

    for (var d = tid; d < headDim; d += 64u) {
        qv[d] = q[h * headDim + d];
    }
    workgroupBarrier();

    for (var t = tid; t < kvLen; t += 64u) {
        var s: f32 = 0.0;
        let kBase = (kvH * params.maxSeqLen + t) * headDim;
        for (var d = 0u; d < headDim; d++) {
            s += qv[d] * kCache[kBase + d];
        }
        scores[t] = s * scale;
    }
    workgroupBarrier();

    var mx: f32 = -1e30;
    for (var t = tid; t < kvLen; t += 64u) {
        mx = max(mx, scores[t]);
    }
    mbuf[tid] = mx;
    workgroupBarrier();
    if (tid == 0u) {
        var m: f32 = -1e30;
        for (var i = 0u; i < 64u; i++) { m = max(m, mbuf[i]); }
        mbuf[0] = m;
    }
    workgroupBarrier();
    let maxScore = mbuf[0];

    var localSum: f32 = 0.0;
    for (var t = tid; t < kvLen; t += 64u) {
        let e = exp(scores[t] - maxScore);
        scores[t] = e;
        localSum += e;
    }
    mbuf[tid] = localSum;
    workgroupBarrier();
    if (tid == 0u) {
        var s: f32 = 0.0;
        for (var i = 0u; i < 64u; i++) { s += mbuf[i]; }
        mbuf[0] = s;
    }
    workgroupBarrier();
    let denom = mbuf[0] + 1e-20;

    for (var d = tid; d < headDim; d += 64u) {
        var acc: f32 = 0.0;
        for (var t = 0u; t < kvLen; t++) {
            let vBase = (kvH * params.maxSeqLen + t) * headDim;
            acc += scores[t] * vCache[vBase + d];
        }
        out[h * headDim + d] = acc / denom;
    }
}
`

// SwiGLU activation: out[i] = silu(gate[i]) * up[i].
const shaderQwenSiLUMul = `
struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read> up: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let g = gate[i];
    out[i] = (g / (1.0 + exp(-g))) * up[i];
}
`

// IncPos: step[0] += 1 (advance decode position).
const shaderQwenIncPos = `
@group(0) @binding(0) var<storage, read_write> step: array<u32>;
@compute @workgroup_size(1)
fn main() { step[0] = step[0] + 1u; }
`
