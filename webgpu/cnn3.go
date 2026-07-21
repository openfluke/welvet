package webgpu

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
	"github.com/openfluke/welvet/core"
)

// CNN3Config holds tiled Conv3d dispatch geometry (cubic kernel: kD=kH=kW=KSize, same stride/padding all axes).
// Layout: input [batch,inC,inD,inH,inW], output [batch,outC,outD,outH,outW], weights [outC,inC*k*k*k].
type CNN3Config struct {
	Batch, InC, InD, InH, InW, OutC, OutD, OutH, OutW, KSize, Stride, Padding int
	MultiCore                                                                 bool
}

const shaderCNN3Scaled = `
struct CNN3ScaleParams {
    batchSize: u32,
    inC: u32, inD: u32, inH: u32, inW: u32,
    outC: u32, outD: u32, outH: u32, outW: u32,
    kD: u32, kH: u32, kW: u32,
    sD: u32, sH: u32, sW: u32,
    pD: u32, pH: u32, pW: u32,
    scale: f32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: CNN3ScaleParams;
@group(0) @binding(1) var<storage, read>       input:   array<f32>;
@group(0) @binding(2) var<storage, read>       weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output:  array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let outIdx_flat = global_id.x;
    let filterIdx   = global_id.y;
    let batchIdx    = global_id.z;

    let oArea = params.outD * params.outH * params.outW;
    if (outIdx_flat >= oArea || filterIdx >= params.outC || batchIdx >= params.batchSize) { return; }

    let outW_pos  = outIdx_flat % params.outW;
    let remainder = outIdx_flat / params.outW;
    let outH_pos  = remainder % params.outH;
    let outD_pos  = remainder / params.outH;

    var sum: f32 = 0.0;
    for (var ic: u32 = 0u; ic < params.inC; ic++) {
        for (var kd: u32 = 0u; kd < params.kD; kd++) {
            for (var kh: u32 = 0u; kh < params.kH; kh++) {
                for (var kw: u32 = 0u; kw < params.kW; kw++) {
                    let inD_pos = i32(outD_pos * params.sD + kd) - i32(params.pD);
                    let inH_pos = i32(outH_pos * params.sH + kh) - i32(params.pH);
                    let inX_pos = i32(outW_pos * params.sW + kw) - i32(params.pW);

                    if (inD_pos >= 0 && u32(inD_pos) < params.inD &&
                        inH_pos >= 0 && u32(inH_pos) < params.inH &&
                        inX_pos >= 0 && u32(inX_pos) < params.inW) {

                        let inIdx = batchIdx * params.inC * params.inD * params.inH * params.inW
                                  + ic * params.inD * params.inH * params.inW
                                  + u32(inD_pos) * params.inH * params.inW
                                  + u32(inH_pos) * params.inW + u32(inX_pos);
                        let wIdx = filterIdx * params.inC * params.kD * params.kH * params.kW
                                 + ic * params.kD * params.kH * params.kW
                                 + kd * params.kH * params.kW
                                 + kh * params.kW + kw;
                        sum += input[inIdx] * weights[wIdx];
                    }
                }
            }
        }
    }
    output[batchIdx * params.outC * oArea + filterIdx * oArea + outIdx_flat] = sum * params.scale;
}
`

func shaderTiledCNN3(tileSize, kernelVol int) string {
	return fmt.Sprintf(`
struct CNN3ScaleParams {
    batchSize: u32,
    inC: u32, inD: u32, inH: u32, inW: u32,
    outC: u32, outD: u32, outH: u32, outW: u32,
    kD: u32, kH: u32, kW: u32,
    sD: u32, sH: u32, sW: u32,
    pD: u32, pH: u32, pW: u32,
    scale: f32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: CNN3ScaleParams;
@group(0) @binding(1) var<storage, read>       input:   array<f32>;
@group(0) @binding(2) var<storage, read>       weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output:  array<f32>;

var<workgroup> wCache: array<f32, %d>;

@compute @workgroup_size(%d, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id:  vec3<u32>,
) {
    let filterIdx = global_id.y;
    let batchIdx  = global_id.z;
    let oArea     = params.outD * params.outH * params.outW;
    let kVol: u32 = %du;
    let wBase     = filterIdx * kVol;

    var i: u32 = local_id.x;
    loop {
        if (i >= kVol) { break; }
        wCache[i] = weights[wBase + i];
        i += %du;
    }
    workgroupBarrier();

    let outIdx_flat = global_id.x;
    if (outIdx_flat >= oArea || filterIdx >= params.outC || batchIdx >= params.batchSize) {
        return;
    }

    let outW_pos  = outIdx_flat %% params.outW;
    let remainder = outIdx_flat / params.outW;
    let outH_pos  = remainder %% params.outH;
    let outD_pos  = remainder / params.outH;

    var sum: f32 = 0.0;
    for (var ic: u32 = 0u; ic < params.inC; ic++) {
        for (var kd: u32 = 0u; kd < params.kD; kd++) {
            for (var kh: u32 = 0u; kh < params.kH; kh++) {
                for (var kw: u32 = 0u; kw < params.kW; kw++) {
                    let inD_pos = i32(outD_pos * params.sD + kd) - i32(params.pD);
                    let inH_pos = i32(outH_pos * params.sH + kh) - i32(params.pH);
                    let inX_pos = i32(outW_pos * params.sW + kw) - i32(params.pW);

                    if (inD_pos >= 0 && u32(inD_pos) < params.inD &&
                        inH_pos >= 0 && u32(inH_pos) < params.inH &&
                        inX_pos >= 0 && u32(inX_pos) < params.inW) {

                        let inIdx = batchIdx * params.inC * params.inD * params.inH * params.inW
                                  + ic * params.inD * params.inH * params.inW
                                  + u32(inD_pos) * params.inH * params.inW
                                  + u32(inH_pos) * params.inW + u32(inX_pos);

                        let cacheIdx = ic * params.kD * params.kH * params.kW
                                     + kd * params.kH * params.kW
                                     + kh * params.kW + kw;

                        sum += input[inIdx] * wCache[cacheIdx];
                    }
                }
            }
        }
    }
    output[batchIdx * params.outC * oArea + filterIdx * oArea + outIdx_flat] = sum * params.scale;
}
`, kernelVol, tileSize, kernelVol, tileSize)
}

type cnn3ScaleParams struct {
	BatchSize             uint32
	InC, InD, InH, InW   uint32
	OutC, OutD, OutH, OutW uint32
	KD, KH, KW            uint32
	SD, SH, SW            uint32
	PD, PH, PW            uint32
	Scale                 float32
	Pad                   uint32
}

type cnn3Bwd3DParams struct {
	BatchSize             uint32
	InC, InD, InH, InW   uint32
	Filters               uint32
	OutD, OutH, OutW      uint32
	KD, KH, KW            uint32
	SD, SH, SW            uint32
	PD, PH, PW            uint32
	Activation            uint32
	Pad                   uint32
}

const wgslCNN3Bwd3DParamsStruct = `
struct CNN3Bwd3DParams {
    batchSize: u32,
    inC: u32, inD: u32, inH: u32, inW: u32,
    filters: u32, outD: u32, outH: u32, outW: u32,
    kD: u32, kH: u32, kW: u32,
    sD: u32, sH: u32, sW: u32,
    pD: u32, pH: u32, pW: u32,
    activation: u32, _pad: u32,
};`

func shaderTiledCNN3BackwardDX(tileSize, kernelVol int) string {
	return fmt.Sprintf(wgslCNN3Bwd3DParamsStruct+`
@group(0) @binding(0) var<uniform>           params:     CNN3Bwd3DParams;
@group(0) @binding(1) var<storage, read>       gradOutput: array<f32>;
@group(0) @binding(2) var<storage, read>       weights:    array<f32>;
@group(0) @binding(3) var<storage, read>       preAct:     array<f32>;
@group(0) @binding(4) var<storage, read_write> gradInput:  array<f32>;

var<workgroup> wCache: array<f32, %d>;
`+cnnWGSLActivateDeriv+`

@compute @workgroup_size(%d, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id:  vec3<u32>,
) {
    let inElemFlat = global_id.x;
    let batchIdx   = global_id.z;
    let kVol: u32  = %du;
    let inDHW      = params.inD * params.inH * params.inW;
    let inVol      = params.inC * inDHW;
    if (batchIdx >= params.batchSize) { return; }

    let ic   = inElemFlat / inDHW;
    let rem1 = inElemFlat %% inDHW;
    let id   = rem1 / (params.inH * params.inW);
    let rem2 = rem1 %% (params.inH * params.inW);
    let ih   = rem2 / params.inW;
    let iw   = rem2 %% params.inW;

    let oArea = params.outD * params.outH * params.outW;
    var sum: f32 = 0.0;

    for (var f: u32 = 0u; f < params.filters; f++) {
        var i: u32 = local_id.x;
        loop {
            if (i >= kVol) { break; }
            wCache[i] = weights[f * kVol + i];
            i += %du;
        }
        workgroupBarrier();

        if (inElemFlat < inVol) {
            for (var kd: u32 = 0u; kd < params.kD; kd++) {
                for (var kh: u32 = 0u; kh < params.kH; kh++) {
                    for (var kw: u32 = 0u; kw < params.kW; kw++) {
                        let vd = i32(id) + i32(params.pD) - i32(kd);
                        let vh = i32(ih) + i32(params.pH) - i32(kh);
                        let vw = i32(iw) + i32(params.pW) - i32(kw);
                        if (vd >= 0 && vd %% i32(params.sD) == 0 &&
                            vh >= 0 && vh %% i32(params.sH) == 0 &&
                            vw >= 0 && vw %% i32(params.sW) == 0) {
                            let od = u32(vd / i32(params.sD));
                            let oh = u32(vh / i32(params.sH));
                            let ow = u32(vw / i32(params.sW));
                            if (od < params.outD && oh < params.outH && ow < params.outW) {
                                let outIdx = (((batchIdx * params.filters + f) * params.outD + od) * params.outH + oh) * params.outW + ow;
                                let dy = gradOutput[outIdx] * activateDerivative(preAct[outIdx], params.activation);
                                let wCacheIdx = ic * params.kD * params.kH * params.kW
                                              + kd * params.kH * params.kW
                                              + kh * params.kW + kw;
                                sum += dy * wCache[wCacheIdx];
                            }
                        }
                    }
                }
            }
        }
        workgroupBarrier();
    }

    if (inElemFlat < inVol) {
        gradInput[batchIdx * inVol + inElemFlat] += sum;
    }
}
`, kernelVol, tileSize, kernelVol, tileSize)
}

func shaderTiledCNN3BackwardDW(tileSize int) string {
	return fmt.Sprintf(wgslCNN3Bwd3DParamsStruct+`
@group(0) @binding(0) var<uniform>           params:      CNN3Bwd3DParams;
@group(0) @binding(1) var<storage, read>       gradOutput:  array<f32>;
@group(0) @binding(2) var<storage, read>       input:       array<f32>;
@group(0) @binding(3) var<storage, read>       preAct:      array<f32>;
@group(0) @binding(4) var<storage, read_write> gradWeights: array<f32>;

var<workgroup> dyCache: array<f32, %d>;
`+cnnWGSLActivateDeriv+`

@compute @workgroup_size(%d, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id:  vec3<u32>,
) {
    let kernelPos = global_id.x;
    let f         = global_id.y;
    let kVol      = params.inC * params.kD * params.kH * params.kW;
    if (f >= params.filters) { return; }

    let kDHW  = params.kD * params.kH * params.kW;
    let kHW   = params.kH * params.kW;
    let ic    = kernelPos / kDHW;
    let kRem  = kernelPos %% kDHW;
    let kd    = kRem / kHW;
    let kRem2 = kRem %% kHW;
    let kh    = kRem2 / params.kW;
    let kw    = kRem2 %% params.kW;

    let oArea        = params.outD * params.outH * params.outW;
    let totalSpatial = params.batchSize * oArea;
    var sum: f32     = 0.0;

    var spatial: u32 = 0u;
    loop {
        if (spatial >= totalSpatial) { break; }

        let loadIdx = spatial + local_id.x;
        if (loadIdx < totalSpatial) {
            let lb      = loadIdx / oArea;
            let lodohow = loadIdx %% oArea;
            let lIdx    = lb * params.filters * oArea + f * oArea + lodohow;
            dyCache[local_id.x] = gradOutput[lIdx] * activateDerivative(preAct[lIdx], params.activation);
        } else {
            dyCache[local_id.x] = 0.0;
        }
        workgroupBarrier();

        if (kernelPos < kVol) {
            for (var ti: u32 = 0u; ti < %du; ti++) {
                let bSpatial = spatial + ti;
                if (bSpatial >= totalSpatial) { break; }
                let b      = bSpatial / oArea;
                let odohow = bSpatial %% oArea;
                let od     = odohow / (params.outH * params.outW);
                let oh     = (odohow / params.outW) %% params.outH;
                let ow     = odohow %% params.outW;

                let id_i = i32(od * params.sD + kd) - i32(params.pD);
                let ih_i = i32(oh * params.sH + kh) - i32(params.pH);
                let iw_i = i32(ow * params.sW + kw) - i32(params.pW);
                if (id_i >= 0 && u32(id_i) < params.inD &&
                    ih_i >= 0 && u32(ih_i) < params.inH &&
                    iw_i >= 0 && u32(iw_i) < params.inW) {
                    let inIdx = (((b * params.inC + ic) * params.inD + u32(id_i)) * params.inH + u32(ih_i)) * params.inW + u32(iw_i);
                    sum += dyCache[ti] * input[inIdx];
                }
            }
        }
        workgroupBarrier();
        spatial += %du;
    }

    if (kernelPos < kVol) {
        gradWeights[f * kVol + kernelPos] += sum;
    }
}
`, tileSize, tileSize, tileSize, tileSize)
}

// CNN3Forward runs on-device Conv3d (pre-activation, scale=1). Layout: [batch,inC,inD,inH,inW] → [batch,outC,outD,outH,outW].
func CNN3Forward(input, weights, output []float32, cfg CNN3Config) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu CNN3Forward: %w", initErr)
	}
	nIn := cfg.Batch * cfg.InC * cfg.InD * cfg.InH * cfg.InW
	nOut := cfg.Batch * cfg.OutC * cfg.OutD * cfg.OutH * cfg.OutW
	nW := cfg.OutC * cfg.InC * cfg.KSize * cfg.KSize * cfg.KSize
	if len(input) < nIn || len(weights) < nW || len(output) < nOut {
		return fmt.Errorf("webgpu CNN3Forward: shape")
	}
	return sess.cnn3Forward(input[:nIn], weights[:nW], output[:nOut], cfg)
}

// CNN3Backward computes gradInput and gradWeights on device (+= into caller-provided zeroed buffers).
func CNN3Backward(gradOut, weights, input, preAct, gradIn, gradW []float32, cfg CNN3Config, act core.ActivationType) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu CNN3Backward: %w", initErr)
	}
	if !CNNTiledBwdOK(act) {
		return fmt.Errorf("webgpu CNN3Backward: unsupported activation %s", act)
	}
	nOut := cfg.Batch * cfg.OutC * cfg.OutD * cfg.OutH * cfg.OutW
	nIn := cfg.Batch * cfg.InC * cfg.InD * cfg.InH * cfg.InW
	nW := cfg.OutC * cfg.InC * cfg.KSize * cfg.KSize * cfg.KSize
	if len(gradOut) < nOut || len(weights) < nW || len(input) < nIn || len(preAct) < nOut ||
		len(gradIn) < nIn || len(gradW) < nW {
		return fmt.Errorf("webgpu CNN3Backward: shape")
	}
	return sess.cnn3Backward(gradOut[:nOut], weights[:nW], input[:nIn], preAct[:nOut],
		gradIn[:nIn], gradW[:nW], cfg, act)
}

func (s *session) cnn3Forward(input, weights, output []float32, cfg CNN3Config) error {
	k := cfg.KSize
	kernelVol := cfg.InC * k * k * k
	tileSize := s.cnnTileSize(cfg.MultiCore)
	tiled := s.cnnUseTiledFwd(kernelVol, tileSize)

	var pipe *wgpu.ComputePipeline
	var err error
	if tiled {
		key := cnnPipeKey(tileSize, kernelVol)
		pipe, err = s.getCNNPipe(&s.cnn3FwdPipes, key, shaderTiledCNN3(tileSize, kernelVol),
			fmt.Sprintf("welvet-cnn3-fwd-%d-%d", tileSize, kernelVol))
	} else {
		key := cnnPipeKey(0, 0)
		pipe, err = s.getCNNPipe(&s.cnn3FwdPipes, key, shaderCNN3Scaled, "welvet-cnn3-fwd-scaled")
	}
	if err != nil {
		return err
	}

	dev, q := s.device, s.queue
	inBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-cnn3-in", Contents: wgpu.ToBytes(input),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer inBuf.Destroy()

	wBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-cnn3-w", Contents: wgpu.ToBytes(weights),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer wBuf.Destroy()

	outBytes := uint64(len(output) * 4)
	if outBytes < 64 {
		outBytes = 64
	}
	outBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "welvet-cnn3-out", Size: outBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer outBuf.Destroy()

	p := cnn3ScaleParams{
		BatchSize: uint32(cfg.Batch),
		InC:       uint32(cfg.InC), InD: uint32(cfg.InD), InH: uint32(cfg.InH), InW: uint32(cfg.InW),
		OutC:      uint32(cfg.OutC), OutD: uint32(cfg.OutD), OutH: uint32(cfg.OutH), OutW: uint32(cfg.OutW),
		KD:        uint32(k), KH: uint32(k), KW: uint32(k),
		SD:        uint32(cfg.Stride), SH: uint32(cfg.Stride), SW: uint32(cfg.Stride),
		PD:        uint32(cfg.Padding), PH: uint32(cfg.Padding), PW: uint32(cfg.Padding),
		Scale:     1,
	}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-cnn3-p", Contents: wgpu.ToBytes([]cnn3ScaleParams{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()

	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: pipe.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: inBuf, Offset: 0, Size: inBuf.GetSize()},
			{Binding: 2, Buffer: wBuf, Offset: 0, Size: wBuf.GetSize()},
			{Binding: 3, Buffer: outBuf, Offset: 0, Size: outBuf.GetSize()},
		},
	})
	if err != nil {
		return err
	}
	defer bg.Release()

	oArea := cfg.OutD * cfg.OutH * cfg.OutW
	enc, err := dev.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipe)
	pass.SetBindGroup(0, bg, nil)
	if tiled {
		pass.DispatchWorkgroups(
			(uint32(oArea)+uint32(tileSize)-1)/uint32(tileSize),
			uint32(cfg.OutC),
			uint32(cfg.Batch),
		)
	} else {
		pass.DispatchWorkgroups(
			(uint32(oArea)+63)/64,
			uint32(cfg.OutC),
			uint32(cfg.Batch),
		)
	}
	pass.End()
	cmd, err := enc.Finish(nil)
	if err != nil {
		return err
	}
	q.Submit(cmd)

	out, err := readbackF32(dev, q, outBuf, len(output))
	if err != nil {
		return err
	}
	copy(output, out)
	return nil
}

func (s *session) cnn3Backward(gradOut, weights, input, preAct, gradIn, gradW []float32, cfg CNN3Config, act core.ActivationType) error {
	k := cfg.KSize
	kernelVol := cfg.InC * k * k * k
	tileSize := s.cnnTileSize(cfg.MultiCore)
	if !s.cnnUseTiledFwd(kernelVol, tileSize) {
		return fmt.Errorf("webgpu CNN3Backward: kernel too large for tiled shared memory")
	}

	keyDX := cnnPipeKey(tileSize, kernelVol)
	pipeDX, err := s.getCNNPipe(&s.cnn3BwdDXPipes, keyDX, shaderTiledCNN3BackwardDX(tileSize, kernelVol),
		fmt.Sprintf("welvet-cnn3-bwd-dx-%d-%d", tileSize, kernelVol))
	if err != nil {
		return err
	}
	keyDW := cnnDWPipeKey(tileSize)
	pipeDW, err := s.getCNNPipe(&s.cnn3BwdDWPipes, keyDW, shaderTiledCNN3BackwardDW(tileSize),
		fmt.Sprintf("welvet-cnn3-bwd-dw-%d", tileSize))
	if err != nil {
		return err
	}

	dev, q := s.device, s.queue
	bwdP := cnn3Bwd3DParams{
		BatchSize: uint32(cfg.Batch),
		InC:       uint32(cfg.InC), InD: uint32(cfg.InD), InH: uint32(cfg.InH), InW: uint32(cfg.InW),
		Filters:   uint32(cfg.OutC),
		OutD:      uint32(cfg.OutD), OutH: uint32(cfg.OutH), OutW: uint32(cfg.OutW),
		KD:        uint32(k), KH: uint32(k), KW: uint32(k),
		SD:        uint32(cfg.Stride), SH: uint32(cfg.Stride), SW: uint32(cfg.Stride),
		PD:        uint32(cfg.Padding), PH: uint32(cfg.Padding), PW: uint32(cfg.Padding),
		Activation: mapCNNActivation(act),
	}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-cnn3-bwd-p", Contents: wgpu.ToBytes([]cnn3Bwd3DParams{bwdP}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()

	gyBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-cnn3-gy", Contents: wgpu.ToBytes(gradOut),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer gyBuf.Destroy()

	wBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-cnn3-w", Contents: wgpu.ToBytes(weights),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer wBuf.Destroy()

	inBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-cnn3-in", Contents: wgpu.ToBytes(input),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer inBuf.Destroy()

	preBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-cnn3-pre", Contents: wgpu.ToBytes(preAct),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer preBuf.Destroy()

	gxBuf, err := zeroF32Buffer(dev, len(gradIn))
	if err != nil {
		return err
	}
	defer gxBuf.Destroy()

	gwBuf, err := zeroF32Buffer(dev, len(gradW))
	if err != nil {
		return err
	}
	defer gwBuf.Destroy()

	bgDX, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: pipeDX.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: gyBuf, Offset: 0, Size: gyBuf.GetSize()},
			{Binding: 2, Buffer: wBuf, Offset: 0, Size: wBuf.GetSize()},
			{Binding: 3, Buffer: preBuf, Offset: 0, Size: preBuf.GetSize()},
			{Binding: 4, Buffer: gxBuf, Offset: 0, Size: gxBuf.GetSize()},
		},
	})
	if err != nil {
		return err
	}
	defer bgDX.Release()

	bgDW, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: pipeDW.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: gyBuf, Offset: 0, Size: gyBuf.GetSize()},
			{Binding: 2, Buffer: inBuf, Offset: 0, Size: inBuf.GetSize()},
			{Binding: 3, Buffer: preBuf, Offset: 0, Size: preBuf.GetSize()},
			{Binding: 4, Buffer: gwBuf, Offset: 0, Size: gwBuf.GetSize()},
		},
	})
	if err != nil {
		return err
	}
	defer bgDW.Release()

	inVol := cfg.InC * cfg.InD * cfg.InH * cfg.InW
	enc, err := dev.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeDX)
	pass.SetBindGroup(0, bgDX, nil)
	pass.DispatchWorkgroups(
		(uint32(inVol)+uint32(tileSize)-1)/uint32(tileSize),
		1,
		uint32(cfg.Batch),
	)
	pass.SetPipeline(pipeDW)
	pass.SetBindGroup(0, bgDW, nil)
	pass.DispatchWorkgroups(
		(uint32(kernelVol)+uint32(tileSize)-1)/uint32(tileSize),
		uint32(cfg.OutC),
		1,
	)
	pass.End()
	cmd, err := enc.Finish(nil)
	if err != nil {
		return err
	}
	q.Submit(cmd)

	gxOut, err := readbackF32(dev, q, gxBuf, len(gradIn))
	if err != nil {
		return err
	}
	copy(gradIn, gxOut)
	gwOut, err := readbackF32(dev, q, gwBuf, len(gradW))
	if err != nil {
		return err
	}
	copy(gradW, gwOut)
	return nil
}
