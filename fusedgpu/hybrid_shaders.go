package fusedgpu

// Fast BinaryG128 hybrid shaders: wg128 tiled GEMV, resid-fused GEMVAdd,
// fused SwiGLU, fused GDN step+gate-norm, on-device argmax.

const shaderBinG128GEMV = `
struct Params { inputSize: u32, outputSize: u32, _p0: u32, _p1: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

const TILE: u32 = 1024u;
var<workgroup> xin: array<f32, 1024>;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let o = wg_id.x * 128u + tid;
    let inN = params.inputSize;
    let outN = params.outputSize;
    var sum: f32 = 0.0;
    var tile: u32 = 0u;
    loop {
        if (tile >= inN) { break; }
        var n = TILE;
        if (tile + n > inN) { n = inN - tile; }
        for (var i = tid; i < n; i += 128u) {
            xin[i] = input[tile + i];
        }
        workgroupBarrier();
        if (o < outN) {
            let wBase = o * inN + tile;
            var c: u32 = 0u;
            // Full 32-weight words (no lim check in hot path).
            loop {
                if (c + 32u > n) { break; }
                let word = weights[(wBase + c) / 32u];
                let scale = scales[(wBase + c) / 128u];
                // 4×8 unroll
                for (var j: u32 = 0u; j < 32u; j += 8u) {
                    let x0 = xin[c + j];
                    let x1 = xin[c + j + 1u];
                    let x2 = xin[c + j + 2u];
                    let x3 = xin[c + j + 3u];
                    let x4 = xin[c + j + 4u];
                    let x5 = xin[c + j + 5u];
                    let x6 = xin[c + j + 6u];
                    let x7 = xin[c + j + 7u];
                    let w = word >> j;
                    sum += x0 * (f32(w & 1u) * 2.0 - 1.0) * scale;
                    sum += x1 * (f32((w >> 1u) & 1u) * 2.0 - 1.0) * scale;
                    sum += x2 * (f32((w >> 2u) & 1u) * 2.0 - 1.0) * scale;
                    sum += x3 * (f32((w >> 3u) & 1u) * 2.0 - 1.0) * scale;
                    sum += x4 * (f32((w >> 4u) & 1u) * 2.0 - 1.0) * scale;
                    sum += x5 * (f32((w >> 5u) & 1u) * 2.0 - 1.0) * scale;
                    sum += x6 * (f32((w >> 6u) & 1u) * 2.0 - 1.0) * scale;
                    sum += x7 * (f32((w >> 7u) & 1u) * 2.0 - 1.0) * scale;
                }
                c += 32u;
            }
            if (c < n) {
                let word = weights[(wBase + c) / 32u];
                let scale = scales[(wBase + c) / 128u];
                for (var j: u32 = 0u; j < n - c; j++) {
                    sum += xin[c + j] * (f32((word >> j) & 1u) * 2.0 - 1.0) * scale;
                }
            }
        }
        workgroupBarrier();
        tile += TILE;
    }
    if (o < outN) { output[o] = sum; }
}
`

// Same GEMV but accumulates into hidden (skips a residual dispatch).
const shaderBinG128GEMVAdd = `
struct Params { inputSize: u32, outputSize: u32, _p0: u32, _p1: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<u32>;
@group(0) @binding(4) var<storage, read_write> hidden: array<f32>;

const TILE: u32 = 1024u;
var<workgroup> xin: array<f32, 1024>;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let o = wg_id.x * 128u + tid;
    let inN = params.inputSize;
    let outN = params.outputSize;
    var sum: f32 = 0.0;
    var tile: u32 = 0u;
    loop {
        if (tile >= inN) { break; }
        var n = TILE;
        if (tile + n > inN) { n = inN - tile; }
        for (var i = tid; i < n; i += 128u) {
            xin[i] = input[tile + i];
        }
        workgroupBarrier();
        if (o < outN) {
            let wBase = o * inN + tile;
            var c: u32 = 0u;
            loop {
                if (c + 32u > n) { break; }
                let word = weights[(wBase + c) / 32u];
                let scale = scales[(wBase + c) / 128u];
                for (var j: u32 = 0u; j < 32u; j += 8u) {
                    let w = word >> j;
                    sum += xin[c + j] * (f32(w & 1u) * 2.0 - 1.0) * scale;
                    sum += xin[c + j + 1u] * (f32((w >> 1u) & 1u) * 2.0 - 1.0) * scale;
                    sum += xin[c + j + 2u] * (f32((w >> 2u) & 1u) * 2.0 - 1.0) * scale;
                    sum += xin[c + j + 3u] * (f32((w >> 3u) & 1u) * 2.0 - 1.0) * scale;
                    sum += xin[c + j + 4u] * (f32((w >> 4u) & 1u) * 2.0 - 1.0) * scale;
                    sum += xin[c + j + 5u] * (f32((w >> 5u) & 1u) * 2.0 - 1.0) * scale;
                    sum += xin[c + j + 6u] * (f32((w >> 6u) & 1u) * 2.0 - 1.0) * scale;
                    sum += xin[c + j + 7u] * (f32((w >> 7u) & 1u) * 2.0 - 1.0) * scale;
                }
                c += 32u;
            }
            if (c < n) {
                let word = weights[(wBase + c) / 32u];
                let scale = scales[(wBase + c) / 128u];
                for (var j: u32 = 0u; j < n - c; j++) {
                    sum += xin[c + j] * (f32((word >> j) & 1u) * 2.0 - 1.0) * scale;
                }
            }
        }
        workgroupBarrier();
        tile += TILE;
    }
    if (o < outN) { hidden[o] = hidden[o] + sum; }
}
`

// Dual Binary GEMV (SwiGLU-style) for same-row projections, e.g. GDN B∥A.
const shaderBinG128Dual = `
struct Params { inputSize: u32, outputSize: u32, _p0: u32, _p1: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> aScales: array<f32>;
@group(0) @binding(3) var<storage, read> aWeights: array<u32>;
@group(0) @binding(4) var<storage, read> bScales: array<f32>;
@group(0) @binding(5) var<storage, read> bWeights: array<u32>;
@group(0) @binding(6) var<storage, read_write> aOut: array<f32>;
@group(0) @binding(7) var<storage, read_write> bOut: array<f32>;

const TILE: u32 = 1024u;
var<workgroup> xin: array<f32, 1024>;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let o = wg_id.x * 128u + tid;
    let inN = params.inputSize;
    let outN = params.outputSize;
    var aSum: f32 = 0.0;
    var bSum: f32 = 0.0;
    var tile: u32 = 0u;
    loop {
        if (tile >= inN) { break; }
        var n = TILE;
        if (tile + n > inN) { n = inN - tile; }
        for (var i = tid; i < n; i += 128u) {
            xin[i] = input[tile + i];
        }
        workgroupBarrier();
        if (o < outN) {
            let wBase = o * inN + tile;
            var c: u32 = 0u;
            loop {
                if (c + 32u > n) { break; }
                let aWord = aWeights[(wBase + c) / 32u];
                let bWord = bWeights[(wBase + c) / 32u];
                let aScale = aScales[(wBase + c) / 128u];
                let bScale = bScales[(wBase + c) / 128u];
                for (var j: u32 = 0u; j < 32u; j += 8u) {
                    let aw = aWord >> j;
                    let bw = bWord >> j;
                    let x0 = xin[c + j];
                    let x1 = xin[c + j + 1u];
                    let x2 = xin[c + j + 2u];
                    let x3 = xin[c + j + 3u];
                    let x4 = xin[c + j + 4u];
                    let x5 = xin[c + j + 5u];
                    let x6 = xin[c + j + 6u];
                    let x7 = xin[c + j + 7u];
                    aSum += x0 * (f32(aw & 1u) * 2.0 - 1.0) * aScale;
                    bSum += x0 * (f32(bw & 1u) * 2.0 - 1.0) * bScale;
                    aSum += x1 * (f32((aw >> 1u) & 1u) * 2.0 - 1.0) * aScale;
                    bSum += x1 * (f32((bw >> 1u) & 1u) * 2.0 - 1.0) * bScale;
                    aSum += x2 * (f32((aw >> 2u) & 1u) * 2.0 - 1.0) * aScale;
                    bSum += x2 * (f32((bw >> 2u) & 1u) * 2.0 - 1.0) * bScale;
                    aSum += x3 * (f32((aw >> 3u) & 1u) * 2.0 - 1.0) * aScale;
                    bSum += x3 * (f32((bw >> 3u) & 1u) * 2.0 - 1.0) * bScale;
                    aSum += x4 * (f32((aw >> 4u) & 1u) * 2.0 - 1.0) * aScale;
                    bSum += x4 * (f32((bw >> 4u) & 1u) * 2.0 - 1.0) * bScale;
                    aSum += x5 * (f32((aw >> 5u) & 1u) * 2.0 - 1.0) * aScale;
                    bSum += x5 * (f32((bw >> 5u) & 1u) * 2.0 - 1.0) * bScale;
                    aSum += x6 * (f32((aw >> 6u) & 1u) * 2.0 - 1.0) * aScale;
                    bSum += x6 * (f32((bw >> 6u) & 1u) * 2.0 - 1.0) * bScale;
                    aSum += x7 * (f32((aw >> 7u) & 1u) * 2.0 - 1.0) * aScale;
                    bSum += x7 * (f32((bw >> 7u) & 1u) * 2.0 - 1.0) * bScale;
                }
                c += 32u;
            }
            if (c < n) {
                let aWord = aWeights[(wBase + c) / 32u];
                let bWord = bWeights[(wBase + c) / 32u];
                let aScale = aScales[(wBase + c) / 128u];
                let bScale = bScales[(wBase + c) / 128u];
                for (var j: u32 = 0u; j < n - c; j++) {
                    let xv = xin[c + j];
                    aSum += xv * (f32((aWord >> j) & 1u) * 2.0 - 1.0) * aScale;
                    bSum += xv * (f32((bWord >> j) & 1u) * 2.0 - 1.0) * bScale;
                }
            }
        }
        workgroupBarrier();
        tile += TILE;
    }
    if (o < outN) {
        aOut[o] = aSum;
        bOut[o] = bSum;
    }
}
`

// Fused BinaryG128 gate+up → silu(gate)*up into intermediate.
const shaderBinG128SwiGLU = `
struct Params { inputSize: u32, intermediate: u32, _p0: u32, _p1: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gateScales: array<f32>;
@group(0) @binding(3) var<storage, read> gateWeights: array<u32>;
@group(0) @binding(4) var<storage, read> upScales: array<f32>;
@group(0) @binding(5) var<storage, read> upWeights: array<u32>;
@group(0) @binding(6) var<storage, read_write> output: array<f32>;

const TILE: u32 = 1024u;
var<workgroup> xin: array<f32, 1024>;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let o = wg_id.x * 128u + tid;
    let inN = params.inputSize;
    let outN = params.intermediate;
    var gSum: f32 = 0.0;
    var uSum: f32 = 0.0;
    var tile: u32 = 0u;
    loop {
        if (tile >= inN) { break; }
        var n = TILE;
        if (tile + n > inN) { n = inN - tile; }
        for (var i = tid; i < n; i += 128u) {
            xin[i] = input[tile + i];
        }
        workgroupBarrier();
        if (o < outN) {
            let wBase = o * inN + tile;
            var c: u32 = 0u;
            loop {
                if (c + 32u > n) { break; }
                let gWord = gateWeights[(wBase + c) / 32u];
                let uWord = upWeights[(wBase + c) / 32u];
                let gScale = gateScales[(wBase + c) / 128u];
                let uScale = upScales[(wBase + c) / 128u];
                for (var j: u32 = 0u; j < 32u; j += 8u) {
                    let gw = gWord >> j;
                    let uw = uWord >> j;
                    let x0 = xin[c + j];
                    let x1 = xin[c + j + 1u];
                    let x2 = xin[c + j + 2u];
                    let x3 = xin[c + j + 3u];
                    let x4 = xin[c + j + 4u];
                    let x5 = xin[c + j + 5u];
                    let x6 = xin[c + j + 6u];
                    let x7 = xin[c + j + 7u];
                    gSum += x0 * (f32(gw & 1u) * 2.0 - 1.0) * gScale;
                    uSum += x0 * (f32(uw & 1u) * 2.0 - 1.0) * uScale;
                    gSum += x1 * (f32((gw >> 1u) & 1u) * 2.0 - 1.0) * gScale;
                    uSum += x1 * (f32((uw >> 1u) & 1u) * 2.0 - 1.0) * uScale;
                    gSum += x2 * (f32((gw >> 2u) & 1u) * 2.0 - 1.0) * gScale;
                    uSum += x2 * (f32((uw >> 2u) & 1u) * 2.0 - 1.0) * uScale;
                    gSum += x3 * (f32((gw >> 3u) & 1u) * 2.0 - 1.0) * gScale;
                    uSum += x3 * (f32((uw >> 3u) & 1u) * 2.0 - 1.0) * uScale;
                    gSum += x4 * (f32((gw >> 4u) & 1u) * 2.0 - 1.0) * gScale;
                    uSum += x4 * (f32((uw >> 4u) & 1u) * 2.0 - 1.0) * uScale;
                    gSum += x5 * (f32((gw >> 5u) & 1u) * 2.0 - 1.0) * gScale;
                    uSum += x5 * (f32((uw >> 5u) & 1u) * 2.0 - 1.0) * uScale;
                    gSum += x6 * (f32((gw >> 6u) & 1u) * 2.0 - 1.0) * gScale;
                    uSum += x6 * (f32((uw >> 6u) & 1u) * 2.0 - 1.0) * uScale;
                    gSum += x7 * (f32((gw >> 7u) & 1u) * 2.0 - 1.0) * gScale;
                    uSum += x7 * (f32((uw >> 7u) & 1u) * 2.0 - 1.0) * uScale;
                }
                c += 32u;
            }
            if (c < n) {
                let gWord = gateWeights[(wBase + c) / 32u];
                let uWord = upWeights[(wBase + c) / 32u];
                let gScale = gateScales[(wBase + c) / 128u];
                let uScale = upScales[(wBase + c) / 128u];
                for (var j: u32 = 0u; j < n - c; j++) {
                    let xv = xin[c + j];
                    gSum += xv * (f32((gWord >> j) & 1u) * 2.0 - 1.0) * gScale;
                    uSum += xv * (f32((uWord >> j) & 1u) * 2.0 - 1.0) * uScale;
                }
            }
        }
        workgroupBarrier();
        tile += TILE;
    }
    if (o < outN) {
        let silu = gSum / (1.0 + exp(-gSum));
        output[o] = silu * uSum;
    }
}
`

const shaderBinEmbed = `
struct Params { hidden: u32, wordsPerRow: u32, groupsPerRow: u32, _p: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> token: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<u32>;
@group(0) @binding(4) var<storage, read_write> hidden: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= params.hidden) { return; }
    let id = token[0];
    let word = weights[id * params.wordsPerRow + c / 32u];
    let scale = scales[id * params.groupsPerRow + c / 128u];
    let bit = (word >> (c % 32u)) & 1u;
    var tw: f32 = -scale;
    if (bit != 0u) { tw = scale; }
    hidden[c] = tw;
}
`

const shaderBinEmbedPrompt = `
struct Params { hidden: u32, wordsPerRow: u32, groupsPerRow: u32, _p: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> step: array<u32>;
@group(0) @binding(2) var<storage, read> prompt: array<u32>;
@group(0) @binding(3) var<storage, read> scales: array<f32>;
@group(0) @binding(4) var<storage, read> weights: array<u32>;
@group(0) @binding(5) var<storage, read_write> hidden: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= params.hidden) { return; }
    let id = prompt[step[0]];
    let word = weights[id * params.wordsPerRow + c / 32u];
    let scale = scales[id * params.groupsPerRow + c / 128u];
    let bit = (word >> (c % 32u)) & 1u;
    var tw: f32 = -scale;
    if (bit != 0u) { tw = scale; }
    hidden[c] = tw;
}
`

const shaderHybridRMS = `
struct Params { dim: u32, eps: f32, _p0: u32, _p1: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let dim = params.dim;
    var local: f32 = 0.0;
    for (var i = tid; i < dim; i += 64u) {
        let v = x[i];
        local += v * v;
    }
    partial[tid] = local;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inv = inverseSqrt(partial[0] / f32(dim) + params.eps);
    for (var i = tid; i < dim; i += 64u) {
        y[i] = x[i] * inv * gamma[i];
    }
}
`

const shaderHybridResid = `
struct Params { dim: u32, _p0: u32, _p1: u32, _p2: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<storage, read> add: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.dim) { return; }
    h[i] = h[i] + add[i];
}
`

const shaderGDNConv = `
struct Params {
    convDim: u32,
    kernel: u32,
    _p0: u32,
    _p1: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> convW: array<f32>;
@group(0) @binding(3) var<storage, read_write> convState: array<f32>;
@group(0) @binding(4) var<storage, read_write> mixed: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ch = gid.x;
    if (ch >= params.convDim) { return; }
    let k = params.kernel;
    let hist = k - 1u;
    var acc: f32 = 0.0;
    let base = ch * hist;
    for (var t: u32 = 0u; t < hist; t++) {
        acc += convW[ch * k + t] * convState[base + t];
    }
    acc += convW[ch * k + (k - 1u)] * qkv[ch];
    mixed[ch] = acc / (1.0 + exp(-acc));
    if (hist > 1u) {
        for (var t: u32 = 0u; t < hist - 1u; t++) {
            convState[base + t] = convState[base + t + 1u];
        }
    }
    if (hist > 0u) {
        convState[base + hist - 1u] = qkv[ch];
    }
}
`

// Fused: L2-norm+repeat key heads AND beta/g for value heads (one dispatch).
const shaderGDNPrepFused = `
struct Params {
    numK: u32,
    numV: u32,
    hdK: u32,
    hdV: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> mixed: array<f32>;
@group(0) @binding(2) var<storage, read_write> qRep: array<f32>;
@group(0) @binding(3) var<storage, read_write> kRep: array<f32>;
@group(0) @binding(4) var<storage, read> betaRaw: array<f32>;
@group(0) @binding(5) var<storage, read> aRaw: array<f32>;
@group(0) @binding(6) var<storage, read> aLog: array<f32>;
@group(0) @binding(7) var<storage, read> dtBias: array<f32>;
@group(0) @binding(8) var<storage, read_write> beta: array<f32>;
@group(0) @binding(9) var<storage, read_write> g: array<f32>;

fn softplus(x: f32) -> f32 {
    if (x > 20.0) { return x; }
    return log(1.0 + exp(x));
}

var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let h = wg_id.x;
    let tid = lid.x;
    let hdK = params.hdK;
    let keyDim = params.numK * hdK;

    if (h < params.numK) {
        let qBase = h * hdK;
        let kBase = keyDim + h * hdK;
        var sq: f32 = 0.0;
        var sk: f32 = 0.0;
        for (var i = tid; i < hdK; i += 64u) {
            let qv = mixed[qBase + i];
            let kv = mixed[kBase + i];
            sq += qv * qv;
            sk += kv * kv;
        }
        red[tid] = sq;
        workgroupBarrier();
        var stride = 32u;
        loop {
            if (stride == 0u) { break; }
            if (tid < stride) { red[tid] += red[tid + stride]; }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let invQ = inverseSqrt(red[0] + 1e-6);
        workgroupBarrier();
        red[tid] = sk;
        workgroupBarrier();
        stride = 32u;
        loop {
            if (stride == 0u) { break; }
            if (tid < stride) { red[tid] += red[tid + stride]; }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let invK = inverseSqrt(red[0] + 1e-6);
        for (var i = tid; i < hdK; i += 64u) {
            mixed[qBase + i] *= invQ;
            mixed[kBase + i] *= invK;
        }
        workgroupBarrier();
        let rep = params.numV / params.numK;
        let scale = inverseSqrt(f32(hdK));
        for (var r: u32 = 0u; r < rep; r++) {
            let dst = (h * rep + r) * hdK;
            for (var i = tid; i < hdK; i += 64u) {
                qRep[dst + i] = mixed[qBase + i] * scale;
                kRep[dst + i] = mixed[kBase + i];
            }
        }
    }

    if (h < params.numV && tid == 0u) {
        beta[h] = 1.0 / (1.0 + exp(-betaRaw[h]));
        let dt = aRaw[h] + dtBias[h];
        g[h] = -exp(aLog[h]) * softplus(dt);
    }
}
`

const shaderGDNStep = `
struct Params {
    numV: u32,
    numK: u32,
    hdK: u32,
    hdV: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read> beta: array<f32>;
@group(0) @binding(5) var<storage, read> g: array<f32>;
@group(0) @binding(6) var<storage, read_write> state: array<f32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;

var<workgroup> scratch: array<f32, 512>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let h = wg_id.x;
    if (h >= params.numV) { return; }
    let tid = lid.x;
    let hdK = params.hdK;
    let hdV = params.hdV;
    let stBase = h * hdK * hdV;
    let gt = exp(g[h]);
    let bt = beta[h];
    let stN = hdK * hdV;

    for (var i = tid; i < stN; i += 64u) {
        state[stBase + i] *= gt;
    }
    workgroupBarrier();

    for (var d = tid; d < hdV; d += 64u) {
        var s: f32 = 0.0;
        for (var j: u32 = 0u; j < hdK; j++) {
            s += state[stBase + j * hdV + d] * k[h * hdK + j];
        }
        scratch[d] = (v[h * hdV + d] - s) * bt;
    }
    workgroupBarrier();

    for (var j = tid; j < hdK; j += 64u) {
        let kj = k[h * hdK + j];
        for (var d: u32 = 0u; d < hdV; d++) {
            state[stBase + j * hdV + d] += kj * scratch[d];
        }
    }
    workgroupBarrier();

    for (var d = tid; d < hdV; d += 64u) {
        var o: f32 = 0.0;
        for (var j: u32 = 0u; j < hdK; j++) {
            o += state[stBase + j * hdV + d] * q[h * hdK + j];
        }
        out[h * hdV + d] = o;
    }
}
`

const shaderGDNGateNorm = `
struct Params { dim: u32, eps: f32, _p0: u32, _p1: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read> z: array<f32>;
@group(0) @binding(3) var<storage, read> gamma: array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let h = wg_id.x;
    let tid = lid.x;
    let hd = params.dim;
    let base = h * hd;
    var local: f32 = 0.0;
    for (var i = tid; i < hd; i += 64u) {
        let v = x[base + i];
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
    let inv = inverseSqrt(partial[0] / f32(hd) + params.eps);
    for (var i = tid; i < hd; i += 64u) {
        let zi = z[base + i];
        let silu = zi / (1.0 + exp(-zi));
        x[base + i] = x[base + i] * inv * gamma[i] * silu;
    }
}
`

const shaderHeadRMS = `
struct Params {
    numHeads: u32,
    headDim: u32,
    epsBits: u32,
    _p: u32,
};
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
    loop {
        if (stride == 0u) { break; }
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

const shaderSplitQGate = `
struct Params {
    numHeads: u32,
    headDim: u32,
    _p0: u32,
    _p1: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> qGate: array<f32>;
@group(0) @binding(2) var<storage, read_write> q: array<f32>;
@group(0) @binding(3) var<storage, read_write> gate: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let nh = params.numHeads;
    let hd = params.headDim;
    let qDim = nh * hd;
    if (i >= qDim) { return; }
    let h = i / hd;
    let d = i % hd;
    let src = h * 2u * hd + d;
    q[i] = qGate[src];
    gate[i] = qGate[src + hd];
}
`

// Qwen3 / Lucy rotate_half inside the rotary prefix (pair d with d+half).
const shaderPartialRoPE = `
struct Params {
    numHeads: u32,
    headDim: u32,
    rotDim: u32,
    thetaBits: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> step: array<u32>;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x;
    if (h >= params.numHeads) { return; }
    let pos = step[0];
    let theta = bitcast<f32>(params.thetaBits);
    let base = h * params.headDim;
    let rot = params.rotDim;
    let half = rot / 2u;
    for (var d: u32 = 0u; d < half; d++) {
        let freq = 1.0 / pow(theta, f32(d * 2u) / f32(rot));
        let ang = f32(pos) * freq;
        let c = cos(ang);
        let s = sin(ang);
        let x0 = x[base + d];
        let x1 = x[base + d + half];
        x[base + d] = x0 * c - x1 * s;
        x[base + d + half] = x0 * s + x1 * c;
    }
}
`

const shaderHybridKVUpdate = `
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

const shaderHybridAttn = `
struct Params {
    numHeads: u32,
    numKVHeads: u32,
    headDim: u32,
    maxSeqLen: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> step: array<u32>;
@group(0) @binding(2) var<storage, read> q: array<f32>;
@group(0) @binding(3) var<storage, read> kCache: array<f32>;
@group(0) @binding(4) var<storage, read> vCache: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

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
        out[h * headDim + d] = acc;
    }
}
`

const shaderOutGate = `
struct Params { dim: u32, _p0: u32, _p1: u32, _p2: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> attn: array<f32>;
@group(0) @binding(2) var<storage, read> gate: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.dim) { return; }
    attn[i] *= 1.0 / (1.0 + exp(-gate[i]));
}
`

const shaderIncPosHybrid = `
@group(0) @binding(0) var<storage, read_write> step: array<u32>;
@compute @workgroup_size(1)
fn main() { step[0] = step[0] + 1u; }
`

const shaderZeroF32 = `
struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> buf: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    buf[i] = 0.0;
}
`

const shaderHybridArgMax = `
struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> outTok: array<u32>;

var<workgroup> bval: array<f32, 256>;
var<workgroup> bidx: array<u32, 256>;

@compute @workgroup_size(256)
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

const shaderGDNStepGNorm = `
struct Params {
    numV: u32,
    hdK: u32,
    hdV: u32,
    epsBits: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read> beta: array<f32>;
@group(0) @binding(5) var<storage, read> g: array<f32>;
@group(0) @binding(6) var<storage, read_write> state: array<f32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<storage, read> z: array<f32>;
@group(0) @binding(9) var<storage, read> gamma: array<f32>;

var<workgroup> scratch: array<f32, 512>;
var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let h = wg_id.x;
    if (h >= params.numV) { return; }
    let tid = lid.x;
    let hdK = params.hdK;
    let hdV = params.hdV;
    let stBase = h * hdK * hdV;
    let gt = exp(g[h]);
    let bt = beta[h];
    let stN = hdK * hdV;

    for (var i = tid; i < stN; i += 64u) {
        state[stBase + i] *= gt;
    }
    workgroupBarrier();

    for (var d = tid; d < hdV; d += 64u) {
        var s: f32 = 0.0;
        for (var j: u32 = 0u; j < hdK; j++) {
            s += state[stBase + j * hdV + d] * k[h * hdK + j];
        }
        scratch[d] = (v[h * hdV + d] - s) * bt;
    }
    workgroupBarrier();

    for (var j = tid; j < hdK; j += 64u) {
        let kj = k[h * hdK + j];
        for (var d: u32 = 0u; d < hdV; d++) {
            state[stBase + j * hdV + d] += kj * scratch[d];
        }
    }
    workgroupBarrier();

    for (var d = tid; d < hdV; d += 64u) {
        var o: f32 = 0.0;
        for (var j: u32 = 0u; j < hdK; j++) {
            o += state[stBase + j * hdV + d] * q[h * hdK + j];
        }
        out[h * hdV + d] = o;
    }
    workgroupBarrier();

    let eps = bitcast<f32>(params.epsBits);
    let base = h * hdV;
    var local: f32 = 0.0;
    for (var i = tid; i < hdV; i += 64u) {
        let vv = out[base + i];
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
    let inv = inverseSqrt(partial[0] / f32(hdV) + eps);
    for (var i = tid; i < hdV; i += 64u) {
        let zi = z[base + i];
        out[base + i] = out[base + i] * inv * gamma[i] * (zi / (1.0 + exp(-zi)));
    }
}
`
