package webgpu

// Shaders for Q4_1, Q5, k-quant, IQ transpose, and any-size Q4/Q8.

const ShaderDenseQ4Any = `
struct Params { batch: u32, inputSize: u32, outputSize: u32, _pad: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<u32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    let b = gid.y;
    if (o >= params.outputSize || b >= params.batch) { return; }
    var sum: f32 = 0.0;
    let xBase = b * params.inputSize;
    let wBase = o * params.inputSize;
    for (var c: u32 = 0u; c < params.inputSize; c++) {
        let flat = wBase + c;
        let blockIdx = flat / 32u;
        let j = flat % 32u;
        let scale = scales[blockIdx];
        let packed = weights[blockIdx * 4u + (j / 8u)];
        let nibble = j % 8u;
        var q = i32((packed >> (nibble * 4u)) & 0xFu);
        if (q > 7) { q -= 16; }
        sum += X[xBase + c] * f32(q) * scale;
    }
    Y[b * params.outputSize + o] = sum;
}
`

const ShaderDenseQ4_1 = `
struct Params { batch: u32, inputSize: u32, outputSize: u32, _pad: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> mins: array<f32>;
@group(0) @binding(4) var<storage, read> weights: array<u32>;
@group(0) @binding(5) var<storage, read_write> Y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    let b = gid.y;
    if (o >= params.outputSize || b >= params.batch) { return; }
    var sum: f32 = 0.0;
    let xBase = b * params.inputSize;
    let wBase = o * params.inputSize;
    for (var c: u32 = 0u; c < params.inputSize; c++) {
        let flat = wBase + c;
        let blockIdx = flat / 32u;
        let j = flat % 32u;
        let scale = scales[blockIdx];
        let mn = mins[blockIdx];
        let packed = weights[blockIdx * 4u + (j / 8u)];
        let nibble = j % 8u;
        let q = (packed >> (nibble * 4u)) & 0xFu;
        sum += X[xBase + c] * (mn + f32(q) * scale);
    }
    Y[b * params.outputSize + o] = sum;
}
`

const ShaderDenseQ5 = `
struct Params {
    batch: u32, inputSize: u32, outputSize: u32, blockBytes: u32,
    hasMin: u32, _p0: u32, _p1: u32, _p2: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> raw: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;

fn load_byte(off: u32) -> u32 {
    let word = raw[off / 4u];
    return (word >> ((off % 4u) * 8u)) & 0xFFu;
}

fn load_f32(off: u32) -> f32 {
    let b0 = load_byte(off);
    let b1 = load_byte(off + 1u);
    let b2 = load_byte(off + 2u);
    let b3 = load_byte(off + 3u);
    return bitcast<f32>(b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u));
}

fn read_bits_at(bitPos: u32, nBits: u32) -> u32 {
    var v: u32 = 0u;
    for (var i: u32 = 0u; i < nBits; i++) {
        let p = bitPos + i;
        let byte = load_byte(p / 8u);
        let bit = (byte >> (p % 8u)) & 1u;
        v |= bit << i;
    }
    return v;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    let b = gid.y;
    if (o >= params.outputSize || b >= params.batch) { return; }
    var sum: f32 = 0.0;
    let xBase = b * params.inputSize;
    let wBase = o * params.inputSize;
    for (var c: u32 = 0u; c < params.inputSize; c++) {
        let flat = wBase + c;
        let bi = flat / 32u;
        let j = flat % 32u;
        let off = bi * params.blockBytes;
        let scale = load_f32(off);
        var w: f32;
        if (params.hasMin != 0u) {
            let mn = load_f32(off + 4u);
            let q = read_bits_at((off + 8u) * 8u + j * 5u, 5u);
            w = mn + f32(q) * scale;
        } else {
            let q = read_bits_at((off + 4u) * 8u + j * 5u, 5u);
            w = f32(i32(q) - 16) * scale;
        }
        sum += X[xBase + c] * w;
    }
    Y[b * params.outputSize + o] = sum;
}
`

const ShaderDenseK = `
struct Params {
    batch: u32, inputSize: u32, outputSize: u32, sbBytes: u32,
    bits: u32, hasDmin: u32, mid: u32, _pad: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> raw: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;

fn load_byte(off: u32) -> u32 {
    let word = raw[off / 4u];
    return (word >> ((off % 4u) * 8u)) & 0xFFu;
}
fn load_f32(off: u32) -> f32 {
    let b0 = load_byte(off);
    let b1 = load_byte(off + 1u);
    let b2 = load_byte(off + 2u);
    let b3 = load_byte(off + 3u);
    return bitcast<f32>(b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u));
}
fn read_bits_at(bitPos: u32, nBits: u32) -> u32 {
    var v: u32 = 0u;
    for (var i: u32 = 0u; i < nBits; i++) {
        let p = bitPos + i;
        let byte = load_byte(p / 8u);
        let bit = (byte >> (p % 8u)) & 1u;
        v |= bit << i;
    }
    return v;
}

fn decode_k(flat: u32) -> f32 {
    let si = flat / 256u;
    let local = flat % 256u;
    let off = si * params.sbBytes;
    let d = load_f32(off);
    let dmin = load_f32(off + 4u);
    let g = local / 16u;
    let scaleU = load_byte(off + 8u + g);
    var sc = d * f32(scaleU) / 255.0;
    if (sc == 0.0) { sc = d / 255.0; }
    var qsOff = off + 8u + 16u;
    var minU: u32 = 0u;
    if (params.hasDmin != 0u) {
        minU = load_byte(off + 8u + 16u + g);
        qsOff = off + 8u + 32u;
    }
    let q = read_bits_at(qsOff * 8u + local * params.bits, params.bits);
    if (params.hasDmin != 0u) {
        var su = i32(minU);
        if (su > 127) { su -= 256; }
        let mn = dmin * f32(su) / 127.0;
        return mn + f32(q) * sc;
    }
    return sc * (f32(q) - f32(params.mid));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    let b = gid.y;
    if (o >= params.outputSize || b >= params.batch) { return; }
    var sum: f32 = 0.0;
    let xBase = b * params.inputSize;
    let wBase = o * params.inputSize;
    for (var c: u32 = 0u; c < params.inputSize; c++) {
        sum += X[xBase + c] * decode_k(wBase + c);
    }
    Y[b * params.outputSize + o] = sum;
}
`

const ShaderDenseKT = `
struct Params {
    batch: u32, inputSize: u32, outputSize: u32, sbBytes: u32,
    bits: u32, hasDmin: u32, mid: u32, _pad: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> GY: array<f32>;
@group(0) @binding(2) var<storage, read> raw: array<u32>;
@group(0) @binding(3) var<storage, read_write> GX: array<f32>;

fn load_byte(off: u32) -> u32 {
    let word = raw[off / 4u];
    return (word >> ((off % 4u) * 8u)) & 0xFFu;
}
fn load_f32(off: u32) -> f32 {
    let b0 = load_byte(off);
    let b1 = load_byte(off + 1u);
    let b2 = load_byte(off + 2u);
    let b3 = load_byte(off + 3u);
    return bitcast<f32>(b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u));
}
fn read_bits_at(bitPos: u32, nBits: u32) -> u32 {
    var v: u32 = 0u;
    for (var i: u32 = 0u; i < nBits; i++) {
        let p = bitPos + i;
        let byte = load_byte(p / 8u);
        let bit = (byte >> (p % 8u)) & 1u;
        v |= bit << i;
    }
    return v;
}
fn decode_k(flat: u32) -> f32 {
    let si = flat / 256u;
    let local = flat % 256u;
    let off = si * params.sbBytes;
    let d = load_f32(off);
    let dmin = load_f32(off + 4u);
    let g = local / 16u;
    let scaleU = load_byte(off + 8u + g);
    var sc = d * f32(scaleU) / 255.0;
    if (sc == 0.0) { sc = d / 255.0; }
    var qsOff = off + 8u + 16u;
    var minU: u32 = 0u;
    if (params.hasDmin != 0u) {
        minU = load_byte(off + 8u + 16u + g);
        qsOff = off + 8u + 32u;
    }
    let q = read_bits_at(qsOff * 8u + local * params.bits, params.bits);
    if (params.hasDmin != 0u) {
        var su = i32(minU);
        if (su > 127) { su -= 256; }
        let mn = dmin * f32(su) / 127.0;
        return mn + f32(q) * sc;
    }
    return sc * (f32(q) - f32(params.mid));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let b = gid.y;
    if (i >= params.inputSize || b >= params.batch) { return; }
    var sum: f32 = 0.0;
    let gyBase = b * params.outputSize;
    for (var o: u32 = 0u; o < params.outputSize; o++) {
        sum += GY[gyBase + o] * decode_k(o * params.inputSize + i);
    }
    GX[b * params.inputSize + i] = sum;
}
`

const ShaderDenseIQT = `
struct Params {
    batch: u32, inputSize: u32, outputSize: u32, bits: u32,
    scaleGroup: u32, nonlinear: u32, midBits: u32, _pad: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> GY: array<f32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> raw: array<u32>;
@group(0) @binding(4) var<storage, read_write> GX: array<f32>;

const IQ4NL: array<f32, 16> = array<f32, 16>(
    -1.0, -0.6961928, -0.52507305, -0.3949175,
    -0.28444138, -0.18477343, -0.091050036, 0.0,
    0.0795803, 0.1609302, 0.2461123, 0.33791524,
    0.44070983, 0.562617, 0.72295684, 1.0
);

fn read_bits(bitPos: u32, nBits: u32) -> u32 {
    var v: u32 = 0u;
    for (var i: u32 = 0u; i < nBits; i++) {
        let p = bitPos + i;
        let word = raw[p / 32u];
        let bit = (word >> (p % 32u)) & 1u;
        v |= bit << i;
    }
    return v;
}
fn dequant(q: u32, scale: f32) -> f32 {
    if (params.nonlinear != 0u) { return scale * IQ4NL[q & 15u]; }
    if (params.bits == 1u) {
        if ((q & 1u) != 0u) { return scale; }
        return -scale;
    }
    let mid = bitcast<f32>(params.midBits);
    return scale * (f32(q) - mid);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let b = gid.y;
    if (i >= params.inputSize || b >= params.batch) { return; }
    var sum: f32 = 0.0;
    let gyBase = b * params.outputSize;
    for (var o: u32 = 0u; o < params.outputSize; o++) {
        let flat = o * params.inputSize + i;
        let sIdx = flat / params.scaleGroup;
        let j = flat % params.scaleGroup;
        let bitPos = sIdx * params.scaleGroup * params.bits + j * params.bits;
        let q = read_bits(bitPos, params.bits);
        sum += GY[gyBase + o] * dequant(q, scales[sIdx]);
    }
    GX[b * params.inputSize + i] = sum;
}
`
