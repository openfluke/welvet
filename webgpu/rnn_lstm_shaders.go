package webgpu

import "fmt"

// ShaderRNNStepPreAct — one RNN timestep: writes linear pre-activation and tanh(h).
const ShaderRNNStepPreAct = `
struct RNNParams {
    batchSize: u32,
    inputSize: u32,
    hiddenSize: u32,
};

@group(0) @binding(0) var<uniform> params: RNNParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> hPrev: array<f32>;
@group(0) @binding(3) var<storage, read> wIH: array<f32>;
@group(0) @binding(4) var<storage, read> wHH: array<f32>;
@group(0) @binding(5) var<storage, read> bias: array<f32>;
@group(0) @binding(6) var<storage, read_write> preAct: array<f32>;
@group(0) @binding(7) var<storage, read_write> hCurr: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let h = global_id.x;
    let b = global_id.y;
    if (h >= params.hiddenSize || b >= params.batchSize) { return; }

    var sum: f32 = bias[h];

    let base_in = b * params.inputSize;
    let base_w_ih = h * params.inputSize;
    for (var i: u32 = 0u; i < params.inputSize; i++) {
        sum += input[base_in + i] * wIH[base_w_ih + i];
    }

    let base_h_prev = b * params.hiddenSize;
    let base_w_hh = h * params.hiddenSize;
    for (var i: u32 = 0u; i < params.hiddenSize; i++) {
        sum += hPrev[base_h_prev + i] * wHH[base_w_hh + i];
    }

    preAct[base_h_prev + h] = sum;
    hCurr[base_h_prev + h] = tanh(sum);
}
`

// ShaderLSTMStepPreAct — one LSTM timestep with [iS,fS,gS,oS,cC] preAct cache.
const ShaderLSTMStepPreAct = `
struct LSTMParams {
    batchSize: u32,
    inputSize: u32,
    hiddenSize: u32,
};

@group(0) @binding(0) var<uniform>             params:  LSTMParams;
@group(0) @binding(1) var<storage, read>       input:   array<f32>;
@group(0) @binding(2) var<storage, read>       hPrev:   array<f32>;
@group(0) @binding(3) var<storage, read>       cPrev:   array<f32>;
@group(0) @binding(4) var<storage, read>       weights: array<f32>;
@group(0) @binding(5) var<storage, read_write> hCurr:   array<f32>;
@group(0) @binding(6) var<storage, read_write> cCurr:   array<f32>;
@group(0) @binding(7) var<storage, read_write> preAct:  array<f32>;

fn lstmpa_sigmoid(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let h = global_id.x;
    let b = global_id.y;
    if (h >= params.hiddenSize || b >= params.batchSize) { return; }

    let ihSize   = params.hiddenSize * params.inputSize;
    let hhSize   = params.hiddenSize * params.hiddenSize;
    let gateSize = ihSize + hhSize + params.hiddenSize;

    let wF_off = gateSize;
    let wG_off = 2u * gateSize;
    let wO_off = 3u * gateSize;

    var iSum: f32 = weights[ihSize + hhSize + h];
    var fSum: f32 = weights[wF_off + ihSize + hhSize + h];
    var gSum: f32 = weights[wG_off + ihSize + hhSize + h];
    var oSum: f32 = weights[wO_off + ihSize + hhSize + h];

    let base_in     = b * params.inputSize;
    let base_h_prev = b * params.hiddenSize;

    for (var i: u32 = 0u; i < params.inputSize; i++) {
        let x     = input[base_in + i];
        let w_idx = h * params.inputSize + i;
        iSum += x * weights[w_idx];
        fSum += x * weights[wF_off + w_idx];
        gSum += x * weights[wG_off + w_idx];
        oSum += x * weights[wO_off + w_idx];
    }
    for (var hp: u32 = 0u; hp < params.hiddenSize; hp++) {
        let hv    = hPrev[base_h_prev + hp];
        let w_idx = ihSize + h * params.hiddenSize + hp;
        iSum += hv * weights[w_idx];
        fSum += hv * weights[wF_off + w_idx];
        gSum += hv * weights[wG_off + w_idx];
        oSum += hv * weights[wO_off + w_idx];
    }

    let iG   = lstmpa_sigmoid(iSum);
    let fG   = lstmpa_sigmoid(fSum);
    let gG   = tanh(gSum);
    let oG   = lstmpa_sigmoid(oSum);
    let cell = fG * cPrev[base_h_prev + h] + iG * gG;

    cCurr[base_h_prev + h] = cell;
    hCurr[base_h_prev + h] = oG * tanh(cell);

    let pIdx = b * 5u * params.hiddenSize;
    preAct[pIdx + h]                          = iSum;
    preAct[pIdx + params.hiddenSize + h]      = fSum;
    preAct[pIdx + 2u * params.hiddenSize + h] = gSum;
    preAct[pIdx + 3u * params.hiddenSize + h] = oSum;
    preAct[pIdx + 4u * params.hiddenSize + h] = cell;
}
`

// shaderTiledRNNBackwardDX — gradInput = gPre @ wIH^T for one RNN step.
func shaderTiledRNNBackwardDX(tileSize int) string {
	return fmt.Sprintf(`
struct RNNParams {
    batchSize:  u32,
    inputSize:  u32,
    hiddenSize: u32,
    padding:    u32,
};
@group(0) @binding(0) var<uniform>             params:     RNNParams;
@group(0) @binding(1) var<storage, read>       gradOutput: array<f32>;
@group(0) @binding(2) var<storage, read>       wIH:        array<f32>;
@group(0) @binding(3) var<storage, read>       hCurr:      array<f32>;
@group(0) @binding(4) var<storage, read_write> gradInput:  array<f32>;

var<workgroup> shGPre: array<f32, %d>;

@compute @workgroup_size(%d, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id)  local_id:  vec3<u32>,
    @builtin(workgroup_id)         wg_id:     vec3<u32>,
) {
    let i   = global_id.x;
    let b   = wg_id.y;
    let tid = local_id.x;
    let H   = params.hiddenSize;
    let I   = params.inputSize;
    let TS: u32 = %du;

    var grad: f32 = 0.0;

    for (var hTile: u32 = 0u; hTile < H; hTile += TS) {
        let h = hTile + tid;
        if (h < H) {
            let hc = hCurr[b * H + h];
            shGPre[tid] = gradOutput[b * H + h] * (1.0 - hc * hc);
        } else {
            shGPre[tid] = 0.0;
        }
        workgroupBarrier();

        if (i < I) {
            let limit = min(TS, H - hTile);
            for (var k: u32 = 0u; k < limit; k++) {
                grad += wIH[(hTile + k) * I + i] * shGPre[k];
            }
        }
        workgroupBarrier();
    }

    if (i < I) {
        gradInput[b * I + i] = grad;
    }
}
`, tileSize, tileSize, tileSize)
}

// shaderTiledRNNBackwardDW — gradWeights = [gradWIH, gradWHH, gradBias].
func shaderTiledRNNBackwardDW(tileSize int) string {
	return fmt.Sprintf(`
struct RNNParams {
    batchSize:  u32,
    inputSize:  u32,
    hiddenSize: u32,
    padding:    u32,
};
@group(0) @binding(0) var<uniform>             params:      RNNParams;
@group(0) @binding(1) var<storage, read>       gradOutput:  array<f32>;
@group(0) @binding(2) var<storage, read>       input:       array<f32>;
@group(0) @binding(3) var<storage, read>       hCurr:       array<f32>;
@group(0) @binding(4) var<storage, read>       hPrev:       array<f32>;
@group(0) @binding(5) var<storage, read_write> gradWeights: array<f32>;

@compute @workgroup_size(%d, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let h = global_id.x;
    if (h >= params.hiddenSize) { return; }

    let H      = params.hiddenSize;
    let I      = params.inputSize;
    let ihSize = H * I;
    let hhSize = H * H;

    var biasGrad: f32 = 0.0;

    for (var b: u32 = 0u; b < params.batchSize; b++) {
        let hc   = hCurr[b * H + h];
        let gPre = gradOutput[b * H + h] * (1.0 - hc * hc);
        biasGrad += gPre;

        for (var i: u32 = 0u; i < I; i++) {
            gradWeights[h * I + i] += gPre * input[b * I + i];
        }
        for (var hp: u32 = 0u; hp < H; hp++) {
            gradWeights[ihSize + h * H + hp] += gPre * hPrev[b * H + hp];
        }
    }
    gradWeights[ihSize + hhSize + h] += biasGrad;
}
`, tileSize)
}

// shaderTiledLSTMBackwardDX — BPTT-aware LSTM input grad for one step.
func shaderTiledLSTMBackwardDX(tileSize int) string {
	return fmt.Sprintf(`
struct LSTMParams {
    batchSize:  u32,
    inputSize:  u32,
    hiddenSize: u32,
    padding:    u32,
};
@group(0) @binding(0) var<uniform>             params:     LSTMParams;
@group(0) @binding(1) var<storage, read>       gradOut:    array<f32>;
@group(0) @binding(2) var<storage, read>       gradHidden: array<f32>;
@group(0) @binding(3) var<storage, read>       gradCell:   array<f32>;
@group(0) @binding(4) var<storage, read>       cPrev:      array<f32>;
@group(0) @binding(5) var<storage, read>       weights:    array<f32>;
@group(0) @binding(6) var<storage, read>       preAct:     array<f32>;
@group(0) @binding(7) var<storage, read_write> gradInput:  array<f32>;

var<workgroup> shDI: array<f32, %d>;
var<workgroup> shDF: array<f32, %d>;
var<workgroup> shDG: array<f32, %d>;
var<workgroup> shDO: array<f32, %d>;

fn lstm_sigmoid(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }

@compute @workgroup_size(%d, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id)  local_id:  vec3<u32>,
    @builtin(workgroup_id)         wg_id:     vec3<u32>,
) {
    let i   = global_id.x;
    let b   = wg_id.y;
    let tid = local_id.x;
    let H   = params.hiddenSize;
    let I   = params.inputSize;
    let TS: u32 = %du;

    let ihSize   = H * I;
    let gateSize = ihSize + H * H + H;

    var grad: f32 = 0.0;

    for (var hTile: u32 = 0u; hTile < H; hTile += TS) {
        let h = hTile + tid;
        if (h < H) {
            let pIdx = b * 5u * H;
            let iS = preAct[pIdx + h];
            let fS = preAct[pIdx + H + h];
            let gS = preAct[pIdx + 2u * H + h];
            let oS = preAct[pIdx + 3u * H + h];
            let cC = preAct[pIdx + 4u * H + h];

            let iG = lstm_sigmoid(iS);
            let fG = lstm_sigmoid(fS);
            let gG = tanh(gS);
            let oG = lstm_sigmoid(oS);
            let cT = tanh(cC);

            let dh = gradOut[b * H + h] + gradHidden[b * H + h];
            let dc = gradCell[b * H + h] + dh * oG * (1.0 - cT * cT);
            let cP = cPrev[b * H + h];

            shDI[tid] = dc * gG * iG * (1.0 - iG);
            shDF[tid] = dc * cP * fG * (1.0 - fG);
            shDG[tid] = dc * iG * (1.0 - gG * gG);
            shDO[tid] = dh * cT * oG * (1.0 - oG);
        } else {
            shDI[tid] = 0.0;
            shDF[tid] = 0.0;
            shDG[tid] = 0.0;
            shDO[tid] = 0.0;
        }
        workgroupBarrier();

        if (i < I) {
            let limit = min(TS, H - hTile);
            for (var k: u32 = 0u; k < limit; k++) {
                let hh = hTile + k;
                grad += weights[hh * I + i]                         * shDI[k]
                      + weights[gateSize + hh * I + i]              * shDF[k]
                      + weights[2u * gateSize + hh * I + i]         * shDG[k]
                      + weights[3u * gateSize + hh * I + i]         * shDO[k];
            }
        }
        workgroupBarrier();
    }

    if (i < I) {
        gradInput[b * I + i] = grad;
    }
}
`, tileSize, tileSize, tileSize, tileSize, tileSize, tileSize)
}

// shaderTiledLSTMBackwardDW — BPTT-aware LSTM weight grad for one step.
func shaderTiledLSTMBackwardDW(tileSize int) string {
	return fmt.Sprintf(`
struct LSTMParams {
    batchSize:  u32,
    inputSize:  u32,
    hiddenSize: u32,
    padding:    u32,
};
@group(0) @binding(0) var<uniform>             params:      LSTMParams;
@group(0) @binding(1) var<storage, read>       gradOut:     array<f32>;
@group(0) @binding(2) var<storage, read>       gradHidden:  array<f32>;
@group(0) @binding(3) var<storage, read>       gradCell:    array<f32>;
@group(0) @binding(4) var<storage, read>       cPrev:       array<f32>;
@group(0) @binding(5) var<storage, read>       input:       array<f32>;
@group(0) @binding(6) var<storage, read>       preAct:      array<f32>;
@group(0) @binding(7) var<storage, read>       hPrev:       array<f32>;
@group(0) @binding(8) var<storage, read_write> gradWeights: array<f32>;

fn lstm_sigmoid_dw(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }

@compute @workgroup_size(%d, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let h = global_id.x;
    if (h >= params.hiddenSize) { return; }

    let H        = params.hiddenSize;
    let I        = params.inputSize;
    let ihSize   = H * I;
    let hhSize   = H * H;
    let gateSize = ihSize + hhSize + H;

    var gBI: f32 = 0.0;
    var gBF: f32 = 0.0;
    var gBG: f32 = 0.0;
    var gBO: f32 = 0.0;

    for (var b: u32 = 0u; b < params.batchSize; b++) {
        let pIdx = b * 5u * H;
        let iS = preAct[pIdx + h];
        let fS = preAct[pIdx + H + h];
        let gS = preAct[pIdx + 2u * H + h];
        let oS = preAct[pIdx + 3u * H + h];
        let cC = preAct[pIdx + 4u * H + h];

        let iG = lstm_sigmoid_dw(iS);
        let fG = lstm_sigmoid_dw(fS);
        let gG = tanh(gS);
        let oG = lstm_sigmoid_dw(oS);
        let cT = tanh(cC);

        let dh = gradOut[b * H + h] + gradHidden[b * H + h];
        let dc = gradCell[b * H + h] + dh * oG * (1.0 - cT * cT);
        let cP = cPrev[b * H + h];

        let diP = dc * gG * iG * (1.0 - iG);
        let dfP = dc * cP * fG * (1.0 - fG);
        let dgP = dc * iG * (1.0 - gG * gG);
        let doP = dh * cT * oG * (1.0 - oG);

        gBI += diP; gBF += dfP; gBG += dgP; gBO += doP;

        for (var i: u32 = 0u; i < I; i++) {
            let x = input[b * I + i];
            gradWeights[h * I + i]                         += diP * x;
            gradWeights[gateSize + h * I + i]              += dfP * x;
            gradWeights[2u * gateSize + h * I + i]         += dgP * x;
            gradWeights[3u * gateSize + h * I + i]         += doP * x;
        }
        for (var hp: u32 = 0u; hp < H; hp++) {
            let hv = hPrev[b * H + hp];
            gradWeights[ihSize + h * H + hp]                         += diP * hv;
            gradWeights[gateSize + ihSize + h * H + hp]              += dfP * hv;
            gradWeights[2u * gateSize + ihSize + h * H + hp]         += dgP * hv;
            gradWeights[3u * gateSize + ihSize + h * H + hp]         += doP * hv;
        }
    }
    gradWeights[ihSize + hhSize + h]                         += gBI;
    gradWeights[gateSize + ihSize + hhSize + h]              += gBF;
    gradWeights[2u * gateSize + ihSize + hhSize + h]         += gBG;
    gradWeights[3u * gateSize + ihSize + hhSize + h]         += gBO;
}
`, tileSize)
}
