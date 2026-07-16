package fusedgpu

// BinaryG128 + hybrid decoder shaders (activations stay on device).

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

// GDN recurrent step for seq_len=1 after q/k/v/beta/g are prepared on device.
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
    let gt = g[h];
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

const shaderIncPosHybrid = `
@group(0) @binding(0) var<storage, read_write> step: array<u32>;
@compute @workgroup_size(1)
fn main() { step[0] = step[0] + 1u; }
`
