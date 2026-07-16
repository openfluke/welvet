package fusedgpu

// BinaryG128 hybrid decoder shaders — weights + activations stay on device.

const shaderBinG128GEMV = `
struct Params { inputSize: u32, outputSize: u32, _p0: u32, _p1: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    if (o >= params.outputSize) { return; }
    var sum: f32 = 0.0;
    let wBase = o * params.inputSize;
    for (var c: u32 = 0u; c < params.inputSize; c += 32u) {
        let word = weights[(wBase + c) / 32u];
        let scale = scales[(wBase + c) / 128u];
        for (var j: u32 = 0u; j < 32u; j++) {
            var tw: f32 = -scale;
            if (((word >> j) & 1u) != 0u) { tw = scale; }
            sum += input[c + j] * tw;
        }
    }
    output[o] = sum;
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

const shaderHybridRMS = `
struct Params { dim: u32, eps: f32, _p0: u32, _p1: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(1)
fn main() {
    var mean: f32 = 0.0;
    for (var i: u32 = 0u; i < params.dim; i++) {
        let v = x[i];
        mean += v * v;
    }
    mean /= f32(params.dim);
    let inv = inverseSqrt(mean + params.eps);
    for (var i: u32 = 0u; i < params.dim; i++) {
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

const shaderHybridSwiGLU = `
struct Params { inter: u32, _p0: u32, _p1: u32, _p2: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> gate: array<f32>;
@group(0) @binding(2) var<storage, read> up: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.inter) { return; }
    let g = gate[i];
    let silu = g / (1.0 + exp(-g));
    gate[i] = silu * up[i];
}
`

// Depthwise causal conv over qkv + silu; updates conv state in place.
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
    // shift state
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

// L2-norm key heads, repeat to value heads, scale q; write qRep/kRep; leave v in mixed.
const shaderGDNPrepQK = `
struct Params {
    numK: u32,
    numV: u32,
    hdK: u32,
    hdV: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> mixed: array<f32>; // [q|k|v]
@group(0) @binding(2) var<storage, read_write> qRep: array<f32>;
@group(0) @binding(3) var<storage, read_write> kRep: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let hi = gid.x;
    if (hi >= params.numK) { return; }
    let hdK = params.hdK;
    let keyDim = params.numK * hdK;
    let qBase = hi * hdK;
    let kBase = keyDim + hi * hdK;
    // L2 q
    var sq: f32 = 0.0;
    for (var i: u32 = 0u; i < hdK; i++) {
        let v = mixed[qBase + i];
        sq += v * v;
    }
    let invQ = inverseSqrt(sq + 1e-6);
    for (var i: u32 = 0u; i < hdK; i++) {
        mixed[qBase + i] *= invQ;
    }
    // L2 k
    var sk: f32 = 0.0;
    for (var i: u32 = 0u; i < hdK; i++) {
        let v = mixed[kBase + i];
        sk += v * v;
    }
    let invK = inverseSqrt(sk + 1e-6);
    for (var i: u32 = 0u; i < hdK; i++) {
        mixed[kBase + i] *= invK;
    }
    let rep = params.numV / params.numK;
    let scale = inverseSqrt(f32(hdK));
    for (var r: u32 = 0u; r < rep; r++) {
        let dst = (hi * rep + r) * hdK;
        for (var i: u32 = 0u; i < hdK; i++) {
            qRep[dst + i] = mixed[qBase + i] * scale;
            kRep[dst + i] = mixed[kBase + i];
        }
    }
}
`

const shaderGDNBetaG = `
struct Params { numV: u32, _p0: u32, _p1: u32, _p2: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> betaRaw: array<f32>;
@group(0) @binding(2) var<storage, read> aRaw: array<f32>;
@group(0) @binding(3) var<storage, read> aLog: array<f32>;
@group(0) @binding(4) var<storage, read> dtBias: array<f32>;
@group(0) @binding(5) var<storage, read_write> beta: array<f32>;
@group(0) @binding(6) var<storage, read_write> g: array<f32>;

fn softplus(x: f32) -> f32 {
    if (x > 20.0) { return x; }
    return log(1.0 + exp(x));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.numV) { return; }
    beta[i] = 1.0 / (1.0 + exp(-betaRaw[i]));
    let dt = aRaw[i] + dtBias[i];
    g[i] = -exp(aLog[i]) * softplus(dt);
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

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x;
    if (h >= params.numV) { return; }
    let hdK = params.hdK;
    let hdV = params.hdV;
    let stBase = h * hdK * hdV;
    let gt = exp(g[h]);
    let bt = beta[h];
    for (var i: u32 = 0u; i < hdK * hdV; i++) {
        state[stBase + i] *= gt;
    }
    for (var d: u32 = 0u; d < hdV; d++) {
        var s: f32 = 0.0;
        for (var j: u32 = 0u; j < hdK; j++) {
            s += state[stBase + j * hdV + d] * k[h * hdK + j];
        }
        let delta = (v[h * hdV + d] - s) * bt;
        for (var j: u32 = 0u; j < hdK; j++) {
            state[stBase + j * hdV + d] += k[h * hdK + j] * delta;
        }
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

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x;
    let hd = params.dim;
    let base = h * hd;
    var mean: f32 = 0.0;
    for (var i: u32 = 0u; i < hd; i++) {
        let v = x[base + i];
        mean += v * v;
    }
    mean /= f32(hd);
    let inv = inverseSqrt(mean + params.eps);
    for (var i: u32 = 0u; i < hd; i++) {
        let zi = z[base + i];
        let silu = zi / (1.0 + exp(-zi));
        x[base + i] = x[base + i] * inv * gamma[i] * silu;
    }
}
`

// Per-head RMSNorm (q or k).
const shaderHeadRMS = `
struct Params {
    numHeads: u32,
    headDim: u32,
    eps: f32,
    _p: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x;
    if (h >= params.numHeads) { return; }
    let hd = params.headDim;
    let base = h * hd;
    var mean: f32 = 0.0;
    for (var i: u32 = 0u; i < hd; i++) {
        let v = x[base + i];
        mean += v * v;
    }
    mean /= f32(hd);
    let inv = inverseSqrt(mean + params.eps);
    for (var i: u32 = 0u; i < hd; i++) {
        x[base + i] = x[base + i] * inv * gamma[i];
    }
}
`

// Split interleaved per-head [q|gate] → qBuf + gateBuf.
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

// Partial RoPE: rotate first rotDim dims with adjacent pairs (Qwen3.5 style).
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
    for (var i: u32 = 0u; i < rot; i += 2u) {
        let freq = 1.0 / pow(theta, f32(i) / f32(rot));
        let ang = f32(pos) * freq;
        let c = cos(ang);
        let s = sin(ang);
        let u = x[base + i];
        let v = x[base + i + 1u];
        x[base + i] = u * c - v * s;
        x[base + i + 1u] = u * s + v * c;
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

// Zero a buffer (f32) — used to clear GDN/KV state on Reset.
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
