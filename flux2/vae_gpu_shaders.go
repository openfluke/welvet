package flux2

// WGSL for fused AutoencoderKLFlux2 decode (NCHW float32).
// Flat 1D kernels use 2D dispatch when n/workgroup > 65535:
//   let i = gid.y * (65535u * WG) + gid.x;

const shaderVAESilu = `
struct P { n: u32, _p0: u32, _p1: u32, _p2: u32, };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> X: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * 16776960u + gid.x; // 65535 * 256
    if (i >= p.n) { return; }
    let v = X[i];
    X[i] = v / (1.0 + exp(-v));
}
`

const shaderVAEAdd = `
struct P { n: u32, scale: f32, _p1: u32, _p2: u32, };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * 16776960u + gid.x;
    if (i >= p.n) { return; }
    Out[i] = (A[i] + B[i]) / p.scale;
}
`

const shaderVAECopy = `
struct P { n: u32, _p0: u32, _p1: u32, _p2: u32, };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Src: array<f32>;
@group(0) @binding(2) var<storage, read_write> Dst: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * 16776960u + gid.x;
    if (i >= p.n) { return; }
    Dst[i] = Src[i];
}
`

const shaderVAEUpsample2x = `
struct P { c: u32, h: u32, w: u32, _p: u32, };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> In: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let oh = p.h * 2u;
    let ow = p.w * 2u;
    let n = p.c * oh * ow;
    let i = gid.y * 16776960u + gid.x;
    if (i >= n) { return; }
    let hw = oh * ow;
    let oc = i / hw;
    let rem = i % hw;
    let oy = rem / ow;
    let ox = rem % ow;
    let iy = oy / 2u;
    let ix = ox / 2u;
    Out[i] = In[oc * p.h * p.w + iy * p.w + ix];
}
`

const shaderVAEConv = `
struct P {
    out_c: u32, in_c: u32, h: u32, w: u32,
    kh: u32, kw: u32, pad: u32, has_bias: u32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> In: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_h = p.h + 2u * p.pad - p.kh + 1u;
    let out_w = p.w + 2u * p.pad - p.kw + 1u;
    let hw = out_h * out_w;
    let n = p.out_c * hw;
    let idx = gid.y * 16776960u + gid.x;
    if (idx >= n) { return; }
    let oc = idx / hw;
    let rem = idx % hw;
    let oy = rem / out_w;
    let ox = rem % out_w;
    let k_area = p.kh * p.kw;
    var acc: f32 = 0.0;
    let w_base = oc * p.in_c * k_area;
    let in_hw = p.h * p.w;
    for (var ic: u32 = 0u; ic < p.in_c; ic++) {
        let src_base = ic * in_hw;
        let wr = w_base + ic * k_area;
        for (var ky: u32 = 0u; ky < p.kh; ky++) {
            let iy = i32(oy) - i32(p.pad) + i32(ky);
            if (iy < 0 || iy >= i32(p.h)) { continue; }
            for (var kx: u32 = 0u; kx < p.kw; kx++) {
                let ix = i32(ox) - i32(p.pad) + i32(kx);
                if (ix < 0 || ix >= i32(p.w)) { continue; }
                acc += In[src_base + u32(iy) * p.w + u32(ix)] * W[wr + ky * p.kw + kx];
            }
        }
    }
    if (p.has_bias != 0u) {
        acc += Bias[oc];
    }
    Out[idx] = acc;
}
`

const shaderVAEGNStats = `
struct P {
    channels: u32, groups: u32, h: u32, w: u32,
    eps: f32, _p0: u32, _p1: u32, _p2: u32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Stats: array<f32>; // [groups*2] mean, inv_std

var<workgroup> sh_sum: array<f32, 256>;
var<workgroup> sh_sq: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let g = wid.x;
    if (g >= p.groups) { return; }
    let ch_per = p.channels / p.groups;
    let hw = p.h * p.w;
    let n = ch_per * hw;
    var sum: f32 = 0.0;
    var sumsq: f32 = 0.0;
    for (var i = lid.x; i < n; i += 256u) {
        let c = i / hw;
        let s = i % hw;
        let v = X[(g * ch_per + c) * hw + s];
        sum += v;
        sumsq += v * v;
    }
    sh_sum[lid.x] = sum;
    sh_sq[lid.x] = sumsq;
    workgroupBarrier();
    for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            sh_sum[lid.x] += sh_sum[lid.x + stride];
            sh_sq[lid.x] += sh_sq[lid.x + stride];
        }
        workgroupBarrier();
    }
    if (lid.x == 0u) {
        let nf = f32(n);
        let mean = sh_sum[0] / nf;
        var var_ = sh_sq[0] / nf - mean * mean;
        if (var_ < 0.0) { var_ = 0.0; }
        Stats[g * 2u] = mean;
        Stats[g * 2u + 1u] = 1.0 / sqrt(var_ + p.eps);
    }
}
`

const shaderVAEGNApply = `
struct P {
    channels: u32, groups: u32, h: u32, w: u32,
    _p0: u32, _p1: u32, _p2: u32, _p3: u32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> Stats: array<f32>;
@group(0) @binding(3) var<storage, read> Gamma: array<f32>;
@group(0) @binding(4) var<storage, read> Beta: array<f32>;
@group(0) @binding(5) var<storage, read_write> Out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let hw = p.h * p.w;
    let n = p.channels * hw;
    let i = gid.y * 16776960u + gid.x;
    if (i >= n) { return; }
    let c = i / hw;
    let ch_per = p.channels / p.groups;
    let g = c / ch_per;
    let mean = Stats[g * 2u];
    let inv = Stats[g * 2u + 1u];
    Out[i] = (X[i] - mean) * inv * Gamma[c] + Beta[c];
}
`

const shaderVAEGEMV = `
struct P { batch: u32, input_size: u32, output_size: u32, has_bias: u32, };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    let b = gid.y;
    if (o >= p.output_size || b >= p.batch) { return; }
    var sum: f32 = 0.0;
    let x_base = b * p.input_size;
    let w_base = o * p.input_size;
    for (var i: u32 = 0u; i < p.input_size; i++) {
        sum += X[x_base + i] * W[w_base + i];
    }
    if (p.has_bias != 0u) {
        sum += Bias[o];
    }
    Y[b * p.output_size + o] = sum;
}
`

const shaderVAEAttn = `
struct P {
    seq: u32, dim: u32, scale: f32, _p: u32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f32>;
@group(0) @binding(3) var<storage, read> V: array<f32>;
@group(0) @binding(4) var<storage, read_write> Out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.seq) { return; }
    var row_max: f32 = -1e30;
    for (var j: u32 = 0u; j < p.seq; j++) {
        var dot: f32 = 0.0;
        let qi = i * p.dim;
        let kj = j * p.dim;
        for (var t: u32 = 0u; t < p.dim; t++) {
            dot += Q[qi + t] * K[kj + t];
        }
        let s = dot * p.scale;
        if (s > row_max) { row_max = s; }
    }
    var sum: f32 = 0.0;
    for (var t: u32 = 0u; t < p.dim; t++) {
        Out[i * p.dim + t] = 0.0;
    }
    for (var j: u32 = 0u; j < p.seq; j++) {
        var dot: f32 = 0.0;
        let qi = i * p.dim;
        let kj = j * p.dim;
        for (var t: u32 = 0u; t < p.dim; t++) {
            dot += Q[qi + t] * K[kj + t];
        }
        let e = exp(dot * p.scale - row_max);
        sum += e;
        let vj = j * p.dim;
        for (var t: u32 = 0u; t < p.dim; t++) {
            Out[i * p.dim + t] += e * V[vj + t];
        }
    }
    let inv = 1.0 / sum;
    for (var t: u32 = 0u; t < p.dim; t++) {
        Out[i * p.dim + t] *= inv;
    }
}
`

const shaderVAENCHWToTok = `
struct P { c: u32, h: u32, w: u32, _p: u32, };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> In: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let seq = p.h * p.w;
    let n = seq * p.c;
    let i = gid.y * 16776960u + gid.x;
    if (i >= n) { return; }
    let s = i / p.c;
    let c = i % p.c;
    Out[i] = In[c * seq + s];
}
`

const shaderVAETokToNCHW = `
struct P { c: u32, h: u32, w: u32, _p: u32, };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> In: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let seq = p.h * p.w;
    let n = seq * p.c;
    let i = gid.y * 16776960u + gid.x;
    if (i >= n) { return; }
    let s = i / p.c;
    let c = i % p.c;
    Out[c * seq + s] = In[i];
}
`
