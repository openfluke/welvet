package webgpu

const ShaderDenseU8 = `
struct Params {
    batch: u32, inputSize: u32, outputSize: u32, _pad: u32,
    minV: f32, scale: f32, _p0: f32, _p1: f32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;

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
        let packed = W[flat / 4u];
        let q = (packed >> ((flat % 4u) * 8u)) & 0xFFu;
        let w = params.minV + f32(q) * params.scale;
        sum += X[xBase + c] * w;
    }
    Y[b * params.outputSize + o] = sum;
}
`

// ShaderDenseNative — decode f16/bf16/fp8/fp4 from packed u32 words on device.
const ShaderDenseNative = `
struct Params { batch: u32, inputSize: u32, outputSize: u32, kind: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;

fn load_u16(i: u32) -> u32 {
    let word = W[i / 2u];
    if ((i % 2u) == 0u) { return word & 0xFFFFu; }
    return word >> 16u;
}
fn load_u8(i: u32) -> u32 {
    let word = W[i / 4u];
    return (word >> ((i % 4u) * 8u)) & 0xFFu;
}
fn f16_to_f32(h: u32) -> f32 {
    let sign = (h >> 15u) & 1u;
    let exp = (h >> 10u) & 0x1Fu;
    let mant = h & 0x3FFu;
    if (exp == 0u) {
        if (mant == 0u) {
            if (sign == 0u) { return 0.0; }
            return -0.0;
        }
        // denorm
        return bitcast<f32>((sign << 31u) | ((mant << 13u))) * exp2(-24.0);
    }
    if (exp == 31u) {
        return bitcast<f32>((sign << 31u) | 0x7f800000u | (mant << 13u));
    }
    let e = exp + 127u - 15u;
    return bitcast<f32>((sign << 31u) | (e << 23u) | (mant << 13u));
}
fn bf16_to_f32(h: u32) -> f32 {
    return bitcast<f32>(h << 16u);
}
fn fp8e4m3_to_f32(b: u32) -> f32 {
    let sign = (b >> 7u) & 1u;
    let exp = (b >> 3u) & 0xFu;
    let mant = b & 7u;
    if (exp == 0u) {
        if (mant == 0u) {
            if (sign == 0u) { return 0.0; }
            return -0.0;
        }
        return select(1.0, -1.0, sign == 1u) * f32(mant) / 8.0 / 64.0;
    }
    let e = i32(exp) - 7;
    var v = (1.0 + f32(mant) / 8.0) * exp2(f32(e));
    if (sign == 1u) { v = -v; }
    return v;
}
fn fp8e5m2_to_f32(b: u32) -> f32 {
    let sign = (b >> 7u) & 1u;
    let exp = (b >> 2u) & 0x1Fu;
    let mant = b & 3u;
    if (exp == 0u) {
        if (mant == 0u) {
            if (sign == 0u) { return 0.0; }
            return -0.0;
        }
        return select(1.0, -1.0, sign == 1u) * f32(mant) / 4.0 / 16384.0;
    }
    if (exp == 31u) {
        return bitcast<f32>((sign << 31u) | 0x7f800000u);
    }
    let e = i32(exp) - 15;
    var v = (1.0 + f32(mant) / 4.0) * exp2(f32(e));
    if (sign == 1u) { v = -v; }
    return v;
}
fn fp4_to_f32(code: u32) -> f32 {
    // E2M1 table
    let c = code & 15u;
    let vals = array<f32, 16>(
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
    );
    return vals[c];
}
fn weight_at(i: u32) -> f32 {
    switch params.kind {
        case 0u: { return f16_to_f32(load_u16(i)); }
        case 1u: { return bf16_to_f32(load_u16(i)); }
        case 2u: { return fp8e4m3_to_f32(load_u8(i)); }
        case 3u: { return fp8e5m2_to_f32(load_u8(i)); }
        case 4u: {
            let byte = load_u8(i / 2u);
            if ((i % 2u) == 0u) { return fp4_to_f32(byte & 15u); }
            return fp4_to_f32(byte >> 4u);
        }
        default: { return 0.0; }
    }
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
        sum += X[xBase + c] * weight_at(wBase + c);
    }
    Y[b * params.outputSize + o] = sum;
}
`

const ShaderDenseDW = `
struct Params { batch: u32, inputSize: u32, outputSize: u32, _pad: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> GY: array<f32>;
@group(0) @binding(3) var<storage, read_write> DW: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    let i = gid.y;
    if (o >= params.outputSize || i >= params.inputSize) { return; }
    var sum: f32 = 0.0;
    for (var b: u32 = 0u; b < params.batch; b++) {
        sum += X[b * params.inputSize + i] * GY[b * params.outputSize + o];
    }
    DW[o * params.inputSize + i] = sum;
}
`

const ShaderDenseQ4_1T = `
struct Params { batch: u32, inputSize: u32, outputSize: u32, _pad: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> GY: array<f32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> mins: array<f32>;
@group(0) @binding(4) var<storage, read> weights: array<u32>;
@group(0) @binding(5) var<storage, read_write> GX: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let b = gid.y;
    if (i >= params.inputSize || b >= params.batch) { return; }
    var sum: f32 = 0.0;
    let gyBase = b * params.outputSize;
    for (var o: u32 = 0u; o < params.outputSize; o++) {
        let flat = o * params.inputSize + i;
        let blockIdx = flat / 32u;
        let j = flat % 32u;
        let scale = scales[blockIdx];
        let mn = mins[blockIdx];
        let packed = weights[blockIdx * 4u + (j / 8u)];
        let q = (packed >> ((j % 8u) * 4u)) & 0xFu;
        sum += GY[gyBase + o] * (mn + f32(q) * scale);
    }
    GX[b * params.inputSize + i] = sum;
}
`

const ShaderDenseQ5T = `
struct Params {
    batch: u32, inputSize: u32, outputSize: u32, blockBytes: u32,
    hasMin: u32, _p0: u32, _p1: u32, _p2: u32,
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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let b = gid.y;
    if (i >= params.inputSize || b >= params.batch) { return; }
    var sum: f32 = 0.0;
    let gyBase = b * params.outputSize;
    for (var o: u32 = 0u; o < params.outputSize; o++) {
        let flat = o * params.inputSize + i;
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
        sum += GY[gyBase + o] * w;
    }
    GX[b * params.inputSize + i] = sum;
}
`
