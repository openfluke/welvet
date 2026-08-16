package fusedgpu

// Aggressive fusion shaders: cut WebGPU storage barriers on Adreno by collapsing
// full-attention post-proj micro-passes and optional attn⊗gate.

// Q path: optional output-gate split + head RMSNorm + partial RoPE (one WG / head).
const shaderAttnQPrep = `
struct Params {
    numHeads: u32,
    headDim: u32,
    epsBits: u32,
    rotDim: u32,
    thetaBits: u32,
    outputGate: u32,
    _p0: u32,
    _p1: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> step: array<u32>;
@group(0) @binding(2) var<storage, read> qGate: array<f32>;
@group(0) @binding(3) var<storage, read_write> q: array<f32>;
@group(0) @binding(4) var<storage, read_write> gate: array<f32>;
@group(0) @binding(5) var<storage, read> gamma: array<f32>;

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
    let base = h * hd;

    if (params.outputGate != 0u) {
        for (var i = tid; i < hd; i += 64u) {
            let src = h * 2u * hd + i;
            q[base + i] = qGate[src];
            gate[base + i] = qGate[src + hd];
        }
        workgroupBarrier();
    }

    let eps = bitcast<f32>(params.epsBits);
    var local: f32 = 0.0;
    for (var i = tid; i < hd; i += 64u) {
        let v = q[base + i];
        local += v * v;
    }
    partial[tid] = local;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { partial[tid] += partial[tid + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inv = inverseSqrt(partial[0] / f32(hd) + eps);
    for (var i = tid; i < hd; i += 64u) {
        q[base + i] = q[base + i] * inv * gamma[i];
    }
    workgroupBarrier();

    let pos = step[0];
    let theta = bitcast<f32>(params.thetaBits);
    let rot = params.rotDim;
    let half = rot / 2u;
    for (var d = tid; d < half; d += 64u) {
        let freq = 1.0 / pow(theta, f32(d * 2u) / f32(rot));
        let ang = f32(pos) * freq;
        let c = cos(ang);
        let s = sin(ang);
        let x0 = q[base + d];
        let x1 = q[base + d + half];
        q[base + d] = x0 * c - x1 * s;
        q[base + d + half] = x0 * s + x1 * c;
    }
}
`

// Merged Q∥KV prep: gate-split + head RMS + RoPE for Q; K RMS + RoPE + KV cache write.
// One WG per query head; KV path runs when h < numKVHeads (GQA).
const shaderAttnPrep = `
struct Params {
    numHeads: u32,
    numKVHeads: u32,
    headDim: u32,
    epsBits: u32,
    rotDim: u32,
    thetaBits: u32,
    outputGate: u32,
    maxSeqLen: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> step: array<u32>;
@group(0) @binding(2) var<storage, read> qGate: array<f32>;
@group(0) @binding(3) var<storage, read_write> q: array<f32>;
@group(0) @binding(4) var<storage, read_write> gate: array<f32>;
@group(0) @binding(5) var<storage, read> qGamma: array<f32>;
@group(0) @binding(6) var<storage, read_write> k: array<f32>;
@group(0) @binding(7) var<storage, read> v: array<f32>;
@group(0) @binding(8) var<storage, read> kGamma: array<f32>;
@group(0) @binding(9) var<storage, read_write> kCache: array<f32>;
@group(0) @binding(10) var<storage, read_write> vCache: array<f32>;

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
    let base = h * hd;
    let eps = bitcast<f32>(params.epsBits);
    let pos = step[0];
    let theta = bitcast<f32>(params.thetaBits);
    let rot = params.rotDim;
    let half = rot / 2u;

    if (params.outputGate != 0u) {
        for (var i = tid; i < hd; i += 64u) {
            let src = h * 2u * hd + i;
            q[base + i] = qGate[src];
            gate[base + i] = qGate[src + hd];
        }
        workgroupBarrier();
    }

    var local: f32 = 0.0;
    for (var i = tid; i < hd; i += 64u) {
        let vv = q[base + i];
        local += vv * vv;
    }
    partial[tid] = local;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { partial[tid] += partial[tid + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let qInv = inverseSqrt(partial[0] / f32(hd) + eps);
    for (var i = tid; i < hd; i += 64u) {
        q[base + i] = q[base + i] * qInv * qGamma[i];
    }
    workgroupBarrier();

    for (var d = tid; d < half; d += 64u) {
        let freq = 1.0 / pow(theta, f32(d * 2u) / f32(rot));
        let ang = f32(pos) * freq;
        let c = cos(ang);
        let s = sin(ang);
        let x0 = q[base + d];
        let x1 = q[base + d + half];
        q[base + d] = x0 * c - x1 * s;
        q[base + d + half] = x0 * s + x1 * c;
    }

    if (h < params.numKVHeads) {
        workgroupBarrier();
        var kLocal: f32 = 0.0;
        for (var i = tid; i < hd; i += 64u) {
            let vv = k[base + i];
            kLocal += vv * vv;
        }
        partial[tid] = kLocal;
        workgroupBarrier();
        stride = 32u;
        loop {
            if (stride == 0u) { break; }
            if (tid < stride) { partial[tid] += partial[tid + stride]; }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let kInv = inverseSqrt(partial[0] / f32(hd) + eps);
        for (var i = tid; i < hd; i += 64u) {
            k[base + i] = k[base + i] * kInv * kGamma[i];
        }
        workgroupBarrier();
        for (var d = tid; d < half; d += 64u) {
            let freq = 1.0 / pow(theta, f32(d * 2u) / f32(rot));
            let ang = f32(pos) * freq;
            let c = cos(ang);
            let s = sin(ang);
            let x0 = k[base + d];
            let x1 = k[base + d + half];
            k[base + d] = x0 * c - x1 * s;
            k[base + d + half] = x0 * s + x1 * c;
        }
        workgroupBarrier();
        let dstBase = (h * params.maxSeqLen + pos) * hd;
        for (var i = tid; i < hd; i += 64u) {
            kCache[dstBase + i] = k[base + i];
            vCache[dstBase + i] = v[base + i];
        }
    }
}
`

// K path: head RMSNorm + partial RoPE, then write this KV head's K and V into cache.
const shaderAttnKVPrep = `
struct Params {
    numKVHeads: u32,
    headDim: u32,
    epsBits: u32,
    rotDim: u32,
    thetaBits: u32,
    maxSeqLen: u32,
    _p0: u32,
    _p1: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> step: array<u32>;
@group(0) @binding(2) var<storage, read_write> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read> gamma: array<f32>;
@group(0) @binding(5) var<storage, read_write> kCache: array<f32>;
@group(0) @binding(6) var<storage, read_write> vCache: array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let h = wg_id.x;
    if (h >= params.numKVHeads) { return; }
    let tid = lid.x;
    let hd = params.headDim;
    let base = h * hd;

    let eps = bitcast<f32>(params.epsBits);
    var local: f32 = 0.0;
    for (var i = tid; i < hd; i += 64u) {
        let vv = k[base + i];
        local += vv * vv;
    }
    partial[tid] = local;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { partial[tid] += partial[tid + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inv = inverseSqrt(partial[0] / f32(hd) + eps);
    for (var i = tid; i < hd; i += 64u) {
        k[base + i] = k[base + i] * inv * gamma[i];
    }
    workgroupBarrier();

    let pos = step[0];
    let theta = bitcast<f32>(params.thetaBits);
    let rot = params.rotDim;
    let half = rot / 2u;
    for (var d = tid; d < half; d += 64u) {
        let freq = 1.0 / pow(theta, f32(d * 2u) / f32(rot));
        let ang = f32(pos) * freq;
        let c = cos(ang);
        let s = sin(ang);
        let x0 = k[base + d];
        let x1 = k[base + d + half];
        k[base + d] = x0 * c - x1 * s;
        k[base + d + half] = x0 * s + x1 * c;
    }
    workgroupBarrier();

    let dstBase = (h * params.maxSeqLen + pos) * hd;
    for (var i = tid; i < hd; i += 64u) {
        kCache[dstBase + i] = k[base + i];
        vCache[dstBase + i] = v[base + i];
    }
}
`

// Attention + optional output gate (silu) in one dispatch.
const shaderAttnGated = `
struct Params {
    numHeads: u32,
    numKVHeads: u32,
    headDim: u32,
    maxSeqLen: u32,
    doGate: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> step: array<u32>;
@group(0) @binding(2) var<storage, read> q: array<f32>;
@group(0) @binding(3) var<storage, read> kCache: array<f32>;
@group(0) @binding(4) var<storage, read> vCache: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<storage, read> gate: array<f32>;

var<workgroup> qv: array<f32, 256>;
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
            acc += (scores[t] / denom) * vCache[vBase + d];
        }
        let oi = h * headDim + d;
        if (params.doGate != 0u) {
            // Match shaderOutGate: attn *= sigmoid(gate), not SiLU.
            acc = acc * (1.0 / (1.0 + exp(-gate[oi])));
        }
        out[oi] = acc;
    }
}
`
