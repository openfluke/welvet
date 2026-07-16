package webgpu

// FormatNone transpose GEMV shaders: gx[b,i] = Σ_o gy[b,o] * W[o,i] with on-device decode.

const ShaderDenseFP32T = `
struct Params { batch: u32, inputSize: u32, outputSize: u32, _pad: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> GY: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> GX: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let b = gid.y;
    if (i >= params.inputSize || b >= params.batch) { return; }
    var sum: f32 = 0.0;
    let gyBase = b * params.outputSize;
    for (var o: u32 = 0u; o < params.outputSize; o++) {
        sum += GY[gyBase + o] * W[o * params.inputSize + i];
    }
    GX[b * params.inputSize + i] = sum;
}
`

const ShaderDenseI8T = `
struct Params {
    batch: u32, inputSize: u32, outputSize: u32, _pad: u32,
    scale: f32, _p0: f32, _p1: f32, _p2: f32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> GY: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read_write> GX: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let b = gid.y;
    if (i >= params.inputSize || b >= params.batch) { return; }
    var sum: f32 = 0.0;
    let gyBase = b * params.outputSize;
    for (var o: u32 = 0u; o < params.outputSize; o++) {
        let flat = o * params.inputSize + i;
        let packed = W[flat / 4u];
        var q = i32((packed >> ((flat % 4u) * 8u)) & 0xFFu);
        if (q >= 128) { q = q - 256; }
        sum += GY[gyBase + o] * f32(q) * params.scale;
    }
    GX[b * params.inputSize + i] = sum;
}
`

const ShaderDenseU8T = `
struct Params {
    batch: u32, inputSize: u32, outputSize: u32, _pad: u32,
    minV: f32, scale: f32, _p0: f32, _p1: f32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> GY: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read_write> GX: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let b = gid.y;
    if (i >= params.inputSize || b >= params.batch) { return; }
    var sum: f32 = 0.0;
    let gyBase = b * params.outputSize;
    for (var o: u32 = 0u; o < params.outputSize; o++) {
        let flat = o * params.inputSize + i;
        let packed = W[flat / 4u];
        let q = (packed >> ((flat % 4u) * 8u)) & 0xFFu;
        sum += GY[gyBase + o] * (params.minV + f32(q) * params.scale);
    }
    GX[b * params.inputSize + i] = sum;
}
`

// ShaderDenseNativeT — same kinds as ShaderDenseNative, transpose GEMV.
const ShaderDenseNativeT = `
struct Params { batch: u32, inputSize: u32, outputSize: u32, kind: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> GY: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read_write> GX: array<f32>;

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
        return bitcast<f32>((sign << 31u) | ((mant << 13u))) * exp2(-24.0);
    }
    if (exp == 31u) {
        return bitcast<f32>((sign << 31u) | 0x7f800000u | (mant << 13u));
    }
    let e = exp + 127u - 15u;
    return bitcast<f32>((sign << 31u) | (e << 23u) | (mant << 13u));
}
fn bf16_to_f32(h: u32) -> f32 { return bitcast<f32>(h << 16u); }
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
    if (exp == 15u) { return select(448.0, -448.0, sign == 1u); }
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
    let i = gid.x;
    let b = gid.y;
    if (i >= params.inputSize || b >= params.batch) { return; }
    var sum: f32 = 0.0;
    let gyBase = b * params.outputSize;
    for (var o: u32 = 0u; o < params.outputSize; o++) {
        sum += GY[gyBase + o] * weight_at(o * params.inputSize + i);
    }
    GX[b * params.inputSize + i] = sum;
}
`
