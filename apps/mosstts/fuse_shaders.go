package mosstts

// WGSL for resident GPT-2 decode fuse (one compute pass per token).

const shaderFuseGEMVBias = `
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

const shaderFuseLayerNorm = `
struct Params { dim: u32, epsBits: u32, _p0: u32, _p1: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

var<workgroup> partial: array<f32, 64>;
var<workgroup> partialSq: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let dim = params.dim;
    var s: f32 = 0.0;
    var sq: f32 = 0.0;
    for (var i = tid; i < dim; i += 64u) {
        let v = input[i];
        s += v;
        sq += v * v;
    }
    partial[tid] = s;
    partialSq[tid] = sq;
    workgroupBarrier();
    var stride = 32u;
    while (stride > 0u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
            partialSq[tid] += partialSq[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let mean = partial[0] / f32(dim);
    let var_ = partialSq[0] / f32(dim) - mean * mean;
    let eps = bitcast<f32>(params.epsBits);
    let inv = inverseSqrt(var_ + eps);
    for (var i = tid; i < dim; i += 64u) {
        output[i] = (input[i] - mean) * inv * weight[i] + bias[i];
    }
}
`

const shaderFuseResidual = `
struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> addend: array<f32>;
@group(0) @binding(2) var<storage, read_write> inout_x: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    inout_x[i] = inout_x[i] + addend[i];
}
`

const shaderFuseGELU = `
struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let v = x[i];
    let inner = 0.7978845608028654 * (v + 0.044715 * v * v * v);
    x[i] = 0.5 * v * (1.0 + tanh(inner));
}
`

// Interleaved GPT-2 / Moss RoPE (rotate_half + repeat_interleave cos/sin).
const shaderFuseRoPE = `
struct Params {
    numHeads: u32,
    headDim: u32,
    thetaBits: u32,
    _pad: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> step: array<u32>; // step[0]=pos
@group(0) @binding(2) var<storage, read_write> x: array<f32>; // Q or K full hidden

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
        let x0 = x[base + 2u * i];
        let x1 = x[base + 2u * i + 1u];
        // rotate_half → (-x1, x0); out = x*cos + rot*sin
        x[base + 2u * i] = x0 * c + (-x1) * s;
        x[base + 2u * i + 1u] = x1 * c + x0 * s;
    }
}
`

const shaderFuseWriteKV = `
struct Params {
    numHeads: u32,
    headDim: u32,
    maxSeq: u32,
    _pad: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> step: array<u32>;
@group(0) @binding(2) var<storage, read> src: array<f32>; // full hidden
@group(0) @binding(3) var<storage, read_write> cache: array<f32>; // [heads,maxSeq,headDim]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x;
    if (h >= params.numHeads) { return; }
    let pos = step[0];
    let baseSrc = h * params.headDim;
    let baseDst = (h * params.maxSeq + pos) * params.headDim;
    for (var d = 0u; d < params.headDim; d++) {
        cache[baseDst + d] = src[baseSrc + d];
    }
}
`

const shaderFuseAttnDecode = `
struct Params {
    numHeads: u32,
    headDim: u32,
    maxSeq: u32,
    scaleBits: u32, // f32 bits; 0 → 1/sqrt(headDim)
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> step: array<u32>;
@group(0) @binding(2) var<storage, read> q: array<f32>;
@group(0) @binding(3) var<storage, read> kCache: array<f32>;
@group(0) @binding(4) var<storage, read> vCache: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

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
    let maxSeq = params.maxSeq;
    let kvLen = step[0] + 1u;
    var scale = bitcast<f32>(params.scaleBits);
    if (params.scaleBits == 0u) {
        scale = inverseSqrt(f32(headDim));
    }

    for (var d = tid; d < headDim; d += 64u) {
        qv[d] = q[h * headDim + d];
    }
    workgroupBarrier();

    for (var t = tid; t < kvLen; t += 64u) {
        var s: f32 = 0.0;
        let kBase = (h * maxSeq + t) * headDim;
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
    let rowMax = mbuf[0];

    var sum: f32 = 0.0;
    for (var t = tid; t < kvLen; t += 64u) {
        let e = exp(scores[t] - rowMax);
        scores[t] = e;
        sum += e;
    }
    mbuf[tid] = sum;
    workgroupBarrier();
    if (tid == 0u) {
        var s: f32 = 0.0;
        for (var i = 0u; i < 64u; i++) { s += mbuf[i]; }
        mbuf[0] = s;
    }
    workgroupBarrier();
    let inv = 1.0 / mbuf[0];

    for (var d = tid; d < headDim; d += 64u) {
        var acc: f32 = 0.0;
        for (var t = 0u; t < kvLen; t++) {
            let w = scores[t] * inv;
            let vBase = (h * maxSeq + t) * headDim;
            acc += w * vCache[vBase + d];
        }
        out[h * headDim + d] = acc;
    }
}
`

const shaderFuseIncPos = `
@group(0) @binding(0) var<storage, read_write> step: array<u32>;
@compute @workgroup_size(1)
fn main() {
    step[0] = step[0] + 1u;
}
`

const shaderFuseCopySlice = `
struct Params { n: u32, srcOff: u32, _p0: u32, _p1: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    dst[i] = src[params.srcOff + i];
}
`
