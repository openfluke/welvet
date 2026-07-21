package webgpu

import "fmt"

// shaderTiledMHAN generates a tiled MHA forward shader (causal or bidirectional via uniform).
func shaderTiledMHAN(tileSize, headDim int) string {
	kvArraySize := tileSize * headDim
	wgSize := 64
	if headDim > 64 {
		wgSize = 128
	}
	if headDim > 128 {
		wgSize = 256
	}
	return fmt.Sprintf(`
struct Params {
    numHeads: u32,
    numKVHeads: u32,
    headDim: u32,
    seqLen: u32,
    kvOffset: u32,
    maxSeqLen: u32,
    tileSize: u32,
    causal: u32,
    kvLen: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> kCache: array<f32>;
@group(0) @binding(3) var<storage, read> vCache: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

var<workgroup> tile_q: array<f32, %d>;
var<workgroup> tile_k: array<f32, %d>;
var<workgroup> tile_v: array<f32, %d>;

@compute @workgroup_size(%d, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let h = wg_id.x;
    let s = wg_id.y;
    if (h >= params.numHeads || s >= params.seqLen) { return; }

    let tid = local_id.x;
    let headDim = params.headDim;
    let kvGroupSize = params.numHeads / params.numKVHeads;
    let kvH = h / kvGroupSize;
    let currentTotalPos = params.kvOffset + s;

    var totalKLen: u32;
    if (params.causal != 0u) {
        totalKLen = currentTotalPos + 1u;
    } else {
        totalKLen = params.kvLen;
    }

    let scale = 1.0 / sqrt(f32(headDim));

    for (var d: u32 = tid; d < headDim; d += %du) {
        tile_q[d] = q[(s * params.numHeads + h) * headDim + d];
    }
    workgroupBarrier();

    var max_score: f32 = -1e38;
    var denom: f32 = 0.0;
    var local_v_acc: f32 = 0.0;

    for (var kTile: u32 = 0u; kTile < totalKLen; kTile += params.tileSize) {
        let currentKSize = min(params.tileSize, totalKLen - kTile);

        for (var i: u32 = tid; i < currentKSize * headDim; i += %du) {
            let row = i / headDim;
            let col = i %% headDim;
            let kvIdx = (kvH * params.maxSeqLen + (kTile + row)) * params.headDim + col;
            tile_k[i] = kCache[kvIdx];
            tile_v[i] = vCache[kvIdx];
        }
        workgroupBarrier();

        for (var j: u32 = 0u; j < currentKSize; j++) {
            let globalKPos = kTile + j;
            if (params.causal != 0u && globalKPos > currentTotalPos) { continue; }

            var score: f32 = 0.0;
            for (var d: u32 = 0u; d < headDim; d++) {
                score += tile_q[d] * tile_k[j * headDim + d];
            }
            score *= scale;

            let old_max = max_score;
            if (score > max_score) {
                max_score = score;
                let exp_factor = exp(old_max - max_score);
                denom = denom * exp_factor + 1.0;
                if (tid < headDim) {
                    local_v_acc = local_v_acc * exp_factor + tile_v[j * headDim + tid];
                }
            } else {
                let exp_val = exp(score - max_score);
                denom += exp_val;
                if (tid < headDim) {
                    local_v_acc += tile_v[j * headDim + tid] * exp_val;
                }
            }
        }
        workgroupBarrier();
    }

    if (tid < headDim) {
        output[(s * params.numHeads + h) * headDim + tid] = local_v_acc / denom;
    }
}
`, headDim, kvArraySize, kvArraySize, wgSize, wgSize, wgSize)
}
