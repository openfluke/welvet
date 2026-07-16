package webgpu

// ShaderDenseExtT — FormatNone Ext transpose GEMV (same kinds as ShaderDenseExt).
const ShaderDenseExtT = `

struct Params {
    batch: u32, inputSize: u32, outputSize: u32, kind: u32,
    bits: u32, _p0: u32, _p1: u32, _p2: u32,
    minV: f32, scale: f32, _p3: f32, _p4: f32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> GY: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read_write> GX: array<f32>;

fn load_byte(off: u32) -> u32 {
    let word = W[off / 4u];
    return (word >> ((off % 4u) * 8u)) & 0xFFu;
}
fn load_u16_at(byteOff: u32) -> u32 {
    return load_byte(byteOff) | (load_byte(byteOff + 1u) << 8u);
}
fn load_u32_at(byteOff: u32) -> u32 {
    return load_byte(byteOff) | (load_byte(byteOff + 1u) << 8u) |
           (load_byte(byteOff + 2u) << 16u) | (load_byte(byteOff + 3u) << 24u);
}
fn load_f32_at(byteOff: u32) -> f32 {
    return bitcast<f32>(load_u32_at(byteOff));
}
fn load_bits(bitPos: u32, nBits: u32) -> u32 {
    var v: u32 = 0u;
    for (var i: u32 = 0u; i < nBits; i++) {
        let p = bitPos + i;
        let bit = (load_byte(p / 8u) >> (p % 8u)) & 1u;
        v |= bit << i;
    }
    return v;
}
fn nf4_table(code: u32) -> f32 {
    let vals = array<f32, 16>(
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    );
    return vals[code & 15u];
}
fn sign_extend(q: u32, bits: u32) -> i32 {
    let signBit = 1u << (bits - 1u);
    let mask = (1u << bits) - 1u;
    let v = q & mask;
    if ((v & signBit) != 0u) {
        return i32(v) - i32(1u << bits);
    }
    return i32(v);
}
fn weight_at(i: u32) -> f32 {
    let sc = params.scale;
    switch params.kind {
        case 0u: { // Uint4
            let body = 4u + (i / 2u);
            let b = load_byte(body);
            var q: u32;
            if ((i % 2u) == 0u) { q = b & 15u; } else { q = b >> 4u; }
            return params.minV + f32(q) * sc;
        }
        case 1u: { // Uint2
            let body = 4u + (i / 4u);
            let b = load_byte(body);
            let q = (b >> ((i % 4u) * 2u)) & 3u;
            return params.minV + f32(q) * sc;
        }
        case 2u: { // NF4
            let b = load_byte(i / 2u);
            var code: u32;
            if ((i % 2u) == 0u) { code = b & 15u; } else { code = b >> 4u; }
            return nf4_table(code) * sc;
        }
        case 3u: { // signed N-bit
            let q = load_bits(i * params.bits, params.bits);
            return f32(sign_extend(q, params.bits)) * sc;
        }
        case 4u: { // unsigned N-bit affine
            let q = load_bits(32u + i * params.bits, params.bits); // skip 4-byte min
            return params.minV + f32(q) * sc;
        }
        case 5u: { // Int16
            let q = i32(load_u16_at(i * 2u));
            if (q >= 32768) { return f32(q - 65536) * sc; }
            return f32(q) * sc;
        }
        case 6u: { // Int32
            return f32(i32(load_u32_at(i * 4u))) * sc;
        }
        case 7u: { // Int64 — low 32 bits sufficient for our pack range
            return f32(i32(load_u32_at(i * 8u))) * sc;
        }
        case 8u: { // Uint16 affine
            let q = load_u16_at(4u + i * 2u);
            return params.minV + f32(q) * sc;
        }
        case 9u: { // Uint32 affine
            let q = load_u32_at(4u + i * 4u);
            return params.minV + f32(q) * sc;
        }
        case 10u: { // Uint64 affine — low 32
            let q = load_u32_at(4u + i * 8u);
            return params.minV + f32(q) * sc;
        }
        case 11u: { // Float64 → f32
            let lo = load_u32_at(i * 8u);
            let hi = load_u32_at(i * 8u + 4u);
            // Convert f64 bits to f32 via WGSL: reconstruct approx from hi (exponent/mant)
            // Prefer bitcast path: pack into f32 via truncating convert on host-equivalent.
            // Use IEEE: take upper bits as bf16-style then refine — exact via bitcast f32 of converted.
            let bits64_hi = hi;
            let sign = bits64_hi >> 31u;
            let exp64 = (bits64_hi >> 20u) & 0x7FFu;
            let mant_hi = bits64_hi & 0xFFFFFu;
            if (exp64 == 0u) { return 0.0; }
            if (exp64 == 0x7FFu) {
                return bitcast<f32>((sign << 31u) | 0x7f800000u);
            }
            var e32 = i32(exp64) - 1023 + 127;
            if (e32 <= 0) { return 0.0; }
            if (e32 >= 255) {
                return bitcast<f32>((sign << 31u) | 0x7f800000u);
            }
            let mant32 = (mant_hi << 3u) | (lo >> 29u);
            return bitcast<f32>((sign << 31u) | (u32(e32) << 23u) | (mant32 & 0x7FFFFFu));
        }
        case 12u: { // Complex64 real = first f32
            return load_f32_at(i * 8u);
        }
        case 13u: { // Complex128 real = first f64 → f32
            // reuse Float64 path on i*16
            let lo = load_u32_at(i * 16u);
            let hi = load_u32_at(i * 16u + 4u);
            let sign = hi >> 31u;
            let exp64 = (hi >> 20u) & 0x7FFu;
            let mant_hi = hi & 0xFFFFFu;
            if (exp64 == 0u) { return 0.0; }
            if (exp64 == 0x7FFu) {
                return bitcast<f32>((sign << 31u) | 0x7f800000u);
            }
            var e32 = i32(exp64) - 1023 + 127;
            if (e32 <= 0) { return 0.0; }
            if (e32 >= 255) {
                return bitcast<f32>((sign << 31u) | 0x7f800000u);
            }
            let mant32 = (mant_hi << 3u) | (lo >> 29u);
            return bitcast<f32>((sign << 31u) | (u32(e32) << 23u) | (mant32 & 0x7FFFFFu));
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
