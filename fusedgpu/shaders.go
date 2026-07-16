package fusedgpu

// Decode-shaped WGSL — shared-mem Q4 GEMV + on-device pos/token for multi-step chunks.

const shaderQ4GEMV = `
struct Params {
    inputSize: u32,
    outputSize: u32,
    _pad0: u32,
    _pad1: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

var<workgroup> xin: array<f32, 2048>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let inN = params.inputSize;
    for (var i = tid; i < inN; i += 64u) {
        xin[i] = input[i];
    }
    workgroupBarrier();

    let o = wg_id.x * 64u + tid;
    if (o >= params.outputSize) { return; }

    var sum: f32 = 0.0;
    let base_w = o * inN;
    let nBlocks = inN / 32u;
    for (var b = 0u; b < nBlocks; b++) {
        let scale = scales[(base_w / 32u) + b];
        let wBase = (base_w / 8u) + b * 4u;
        let iBase = b * 32u;
        for (var w = 0u; w < 4u; w++) {
            let packed = weights[wBase + w];
            let i0 = iBase + w * 8u;
            var acc: f32 = 0.0;
            for (var n = 0u; n < 8u; n++) {
                var q = i32((packed >> (n * 4u)) & 0xFu);
                if (q > 7) { q -= 16; }
                acc += xin[i0 + n] * f32(q);
            }
            sum += acc * scale;
        }
    }
    output[o] = sum;
}
`

// One shared-memory load of x → Q/K/V into packed buffer [Q|K|V].
const shaderQ4GEMV_QKV = `
struct Params {
    inputSize: u32,
    qDim: u32,
    kvDim: u32,
    _pad: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> qScales: array<f32>;
@group(0) @binding(3) var<storage, read> qWeights: array<u32>;
@group(0) @binding(4) var<storage, read> kScales: array<f32>;
@group(0) @binding(5) var<storage, read> kWeights: array<u32>;
@group(0) @binding(6) var<storage, read> vScales: array<f32>;
@group(0) @binding(7) var<storage, read> vWeights: array<u32>;
@group(0) @binding(8) var<storage, read_write> qkvOut: array<f32>;

var<workgroup> xin: array<f32, 2048>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let inN = params.inputSize;
    for (var i = tid; i < inN; i += 64u) {
        xin[i] = input[i];
    }
    workgroupBarrier();

    let total = params.qDim + params.kvDim + params.kvDim;
    let o = wg_id.x * 64u + tid;
    if (o >= total) { return; }

    var row: u32;
    var sum: f32 = 0.0;
    let nBlocks = inN / 32u;
    if (o < params.qDim) {
        row = o;
        let base_w = row * inN;
        for (var b = 0u; b < nBlocks; b++) {
            let scale = qScales[(base_w / 32u) + b];
            let wBase = (base_w / 8u) + b * 4u;
            let iBase = b * 32u;
            for (var w = 0u; w < 4u; w++) {
                let packed = qWeights[wBase + w];
                let i0 = iBase + w * 8u;
                var acc: f32 = 0.0;
                for (var n = 0u; n < 8u; n++) {
                    var q = i32((packed >> (n * 4u)) & 0xFu);
                    if (q > 7) { q -= 16; }
                    acc += xin[i0 + n] * f32(q);
                }
                sum += acc * scale;
            }
        }
    } else if (o < params.qDim + params.kvDim) {
        row = o - params.qDim;
        let base_w = row * inN;
        for (var b = 0u; b < nBlocks; b++) {
            let scale = kScales[(base_w / 32u) + b];
            let wBase = (base_w / 8u) + b * 4u;
            let iBase = b * 32u;
            for (var w = 0u; w < 4u; w++) {
                let packed = kWeights[wBase + w];
                let i0 = iBase + w * 8u;
                var acc: f32 = 0.0;
                for (var n = 0u; n < 8u; n++) {
                    var q = i32((packed >> (n * 4u)) & 0xFu);
                    if (q > 7) { q -= 16; }
                    acc += xin[i0 + n] * f32(q);
                }
                sum += acc * scale;
            }
        }
    } else {
        row = o - params.qDim - params.kvDim;
        let base_w = row * inN;
        for (var b = 0u; b < nBlocks; b++) {
            let scale = vScales[(base_w / 32u) + b];
            let wBase = (base_w / 8u) + b * 4u;
            let iBase = b * 32u;
            for (var w = 0u; w < 4u; w++) {
                let packed = vWeights[wBase + w];
                let i0 = iBase + w * 8u;
                var acc: f32 = 0.0;
                for (var n = 0u; n < 8u; n++) {
                    var q = i32((packed >> (n * 4u)) & 0xFu);
                    if (q > 7) { q -= 16; }
                    acc += xin[i0 + n] * f32(q);
                }
                sum += acc * scale;
            }
        }
    }
    qkvOut[o] = sum;
}
`

const shaderQ4SwiGLUFused = `
struct Params {
    inputSize: u32,
    intermediate: u32,
    _pad0: u32,
    _pad1: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gateScales: array<f32>;
@group(0) @binding(3) var<storage, read> gateWeights: array<u32>;
@group(0) @binding(4) var<storage, read> upScales: array<f32>;
@group(0) @binding(5) var<storage, read> upWeights: array<u32>;
@group(0) @binding(6) var<storage, read_write> output: array<f32>;

var<workgroup> xin: array<f32, 2048>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let inN = params.inputSize;
    for (var i = tid; i < inN; i += 64u) {
        xin[i] = input[i];
    }
    workgroupBarrier();

    let o = wg_id.x * 64u + tid;
    if (o >= params.intermediate) { return; }

    var g: f32 = 0.0;
    var u: f32 = 0.0;
    let base_w = o * inN;
    let nBlocks = inN / 32u;
    for (var b = 0u; b < nBlocks; b++) {
        let gScale = gateScales[(base_w / 32u) + b];
        let uScale = upScales[(base_w / 32u) + b];
        let wBase = (base_w / 8u) + b * 4u;
        let iBase = b * 32u;
        for (var w = 0u; w < 4u; w++) {
            let gp = gateWeights[wBase + w];
            let up = upWeights[wBase + w];
            let i0 = iBase + w * 8u;
            var gAcc: f32 = 0.0;
            var uAcc: f32 = 0.0;
            for (var n = 0u; n < 8u; n++) {
                var qg = i32((gp >> (n * 4u)) & 0xFu);
                if (qg > 7) { qg -= 16; }
                var qu = i32((up >> (n * 4u)) & 0xFu);
                if (qu > 7) { qu -= 16; }
                let xv = xin[i0 + n];
                gAcc += xv * f32(qg);
                uAcc += xv * f32(qu);
            }
            g += gAcc * gScale;
            u += uAcc * uScale;
        }
    }
    let sig = 1.0 / (1.0 + exp(-g));
    output[o] = (g * sig) * u;
}
`

const shaderRMSNorm = `
struct Params {
    dim: u32,
    eps: f32,
    _pad0: u32,
    _pad1: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
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

const shaderResidual = `
struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> inout_b: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    inout_b[i] = inout_b[i] + a[i];
}
`

// RoPE reads position from step[0] (GPU-resident).
const shaderRoPE = `
struct Params {
    numHeads: u32,
    headDim: u32,
    thetaBits: u32, // float bits
    _pad: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> step: array<u32>; // step[0] = pos
@group(0) @binding(2) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(64, 1, 1)
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

const shaderAttnDecode = `
struct Params {
    numHeads: u32,
    numKVHeads: u32,
    headDim: u32,
    maxSeqLen: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> step: array<u32>; // pos
@group(0) @binding(2) var<storage, read> q: array<f32>;
@group(0) @binding(3) var<storage, read> kCache: array<f32>;
@group(0) @binding(4) var<storage, read> vCache: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

var<workgroup> qv: array<f32, 128>;
var<workgroup> scores: array<f32, 2048>;
var<workgroup> mbuf: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
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

const shaderKVUpdate = `
struct Params {
    kvDim: u32,
    maxSeqLen: u32,
    headDim: u32,
    _pad: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> step: array<u32>;
@group(0) @binding(2) var<storage, read> kIn: array<f32>;
@group(0) @binding(3) var<storage, read> vIn: array<f32>;
@group(0) @binding(4) var<storage, read_write> kCache: array<f32>;
@group(0) @binding(5) var<storage, read_write> vCache: array<f32>;

@compute @workgroup_size(64, 1, 1)
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

const shaderEmbed = `
struct Params {
    hidden: u32,
    vocab: u32,
    _p0: u32,
    _p1: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> token: array<u32>;
@group(0) @binding(2) var<storage, read> table: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    if (d >= params.hidden) { return; }
    let tok = token[0];
    out[d] = table[tok * params.hidden + d];
}
`

const shaderEmbedPrompt = `
struct Params {
    hidden: u32,
    vocab: u32,
    _p0: u32,
    _p1: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> step: array<u32>;
@group(0) @binding(2) var<storage, read> prompt: array<u32>;
@group(0) @binding(3) var<storage, read> table: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    if (d >= params.hidden) { return; }
    let tok = prompt[step[0]];
    out[d] = table[tok * params.hidden + d];
}
`

const shaderArgMax = `
struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> outTok: array<u32>;

var<workgroup> bval: array<f32, 256>;
var<workgroup> bidx: array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    var bestV: f32 = -3.402823466e+38;
    var bestI: u32 = 0u;
    var i = lane;
    loop {
        if (i >= params.n) { break; }
        let v = logits[i];
        if (v > bestV) { bestV = v; bestI = i; }
        i += 256u;
    }
    bval[lane] = bestV;
    bidx[lane] = bestI;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride) {
            if (bval[lane + stride] > bval[lane]) {
                bval[lane] = bval[lane + stride];
                bidx[lane] = bidx[lane + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lane == 0u) { outTok[0] = bidx[0]; }
}
`

// step[0]=pos, step[1]=outCount. After sample: hist[outCount]=tok; token=tok; pos++; outCount++.
const shaderAdvance = `
@group(0) @binding(0) var<storage, read_write> step: array<u32>;
@group(0) @binding(1) var<storage, read> outTok: array<u32>;
@group(0) @binding(2) var<storage, read_write> history: array<u32>;
@group(0) @binding(3) var<storage, read_write> token: array<u32>;

@compute @workgroup_size(1, 1, 1)
fn main() {
    let pos = step[0];
    let oc = step[1];
    let tok = outTok[0];
    history[oc] = tok;
    token[0] = tok;
    step[0] = pos + 1u;
    step[1] = oc + 1u;
}
`

const shaderIncPos = `
@group(0) @binding(0) var<storage, read_write> step: array<u32>;
@compute @workgroup_size(1, 1, 1)
fn main() {
    step[0] = step[0] + 1u;
}
`
