package webgpu

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
	"github.com/openfluke/welvet/core"
)

// CNN2Config holds tiled Conv2d dispatch geometry (square kernel: kH=kW=KSize, sH=sW=Stride, pH=pW=Padding).
// Layout: input [batch,inC,inH,inW], output [batch,outC,outH,outW], weights [outC,inC*k*k].
type CNN2Config struct {
	Batch, InC, InH, InW, OutC, OutH, OutW, KSize, Stride, Padding int
	MultiCore                                                       bool
}

const shaderCNN2Scaled = `
struct CNN2ScaleParams {
    batchSize: u32,
    inC: u32, inH: u32, inW: u32,
    outC: u32, outH: u32, outW: u32,
    kH: u32, kW: u32,
    sH: u32, sW: u32,
    pH: u32, pW: u32,
    scale: f32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform>             params:  CNN2ScaleParams;
@group(0) @binding(1) var<storage, read>       input:   array<f32>;
@group(0) @binding(2) var<storage, read>       weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output:  array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let outFlat   = global_id.x;
    let filterIdx = global_id.y;
    let batchIdx  = global_id.z;
    let oArea = params.outH * params.outW;
    if (outFlat >= oArea || filterIdx >= params.outC || batchIdx >= params.batchSize) { return; }

    let oh = outFlat / params.outW;
    let ow = outFlat % params.outW;
    var sum: f32 = 0.0;

    for (var ic: u32 = 0u; ic < params.inC; ic++) {
        for (var kh: u32 = 0u; kh < params.kH; kh++) {
            for (var kw: u32 = 0u; kw < params.kW; kw++) {
                let ih = i32(oh * params.sH + kh) - i32(params.pH);
                let iw = i32(ow * params.sW + kw) - i32(params.pW);
                if (ih >= 0 && u32(ih) < params.inH && iw >= 0 && u32(iw) < params.inW) {
                    let inIdx = batchIdx * params.inC * params.inH * params.inW
                              + ic * params.inH * params.inW
                              + u32(ih) * params.inW + u32(iw);
                    let wIdx = filterIdx * params.inC * params.kH * params.kW
                             + ic * params.kH * params.kW
                             + kh * params.kW + kw;
                    sum += input[inIdx] * weights[wIdx];
                }
            }
        }
    }
    output[batchIdx * params.outC * oArea + filterIdx * oArea + outFlat] = sum * params.scale;
}
`

func shaderTiledCNN2(tileSize, kernelVol int) string {
	return fmt.Sprintf(`
struct CNN2ScaleParams {
    batchSize: u32,
    inC: u32, inH: u32, inW: u32,
    outC: u32, outH: u32, outW: u32,
    kH: u32, kW: u32,
    sH: u32, sW: u32,
    pH: u32, pW: u32,
    scale: f32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform>             params:  CNN2ScaleParams;
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
    let oArea     = params.outH * params.outW;
    let kVol: u32 = %du;
    let wBase     = filterIdx * kVol;

    var i: u32 = local_id.x;
    loop {
        if (i >= kVol) { break; }
        wCache[i] = weights[wBase + i];
        i += %du;
    }
    workgroupBarrier();

    let outFlat = global_id.x;
    if (outFlat >= oArea || filterIdx >= params.outC || batchIdx >= params.batchSize) { return; }

    let oh = outFlat / params.outW;
    let ow = outFlat %% params.outW;
    var sum: f32 = 0.0;

    for (var ic: u32 = 0u; ic < params.inC; ic++) {
        for (var kh: u32 = 0u; kh < params.kH; kh++) {
            for (var kw: u32 = 0u; kw < params.kW; kw++) {
                let ih = i32(oh * params.sH + kh) - i32(params.pH);
                let iw = i32(ow * params.sW + kw) - i32(params.pW);
                if (ih >= 0 && u32(ih) < params.inH && iw >= 0 && u32(iw) < params.inW) {
                    let inIdx = batchIdx * params.inC * params.inH * params.inW
                              + ic * params.inH * params.inW
                              + u32(ih) * params.inW + u32(iw);
                    let cacheIdx = ic * params.kH * params.kW + kh * params.kW + kw;
                    sum += input[inIdx] * wCache[cacheIdx];
                }
            }
        }
    }
    output[batchIdx * params.outC * oArea + filterIdx * oArea + outFlat] = sum * params.scale;
}
`, kernelVol, tileSize, kernelVol, tileSize)
}

type cnn2ScaleParams struct {
	BatchSize        uint32
	InC, InH, InW   uint32
	OutC, OutH, OutW uint32
	KH, KW          uint32
	SH, SW          uint32
	PH, PW          uint32
	Scale           float32
	Pad             uint32
}

type cnn2Bwd2DParams struct {
	BatchSize        uint32
	InC, InH, InW   uint32
	Filters          uint32
	OutH, OutW      uint32
	KH, KW          uint32
	SH, SW          uint32
	PH, PW          uint32
	Activation      uint32
	Pad             uint32
}

const wgslCNN2Bwd2DParamsStruct = `
struct CNN2Bwd2DParams {
    batchSize: u32,
    inC: u32, inH: u32, inW: u32,
    filters: u32, outH: u32, outW: u32,
    kH: u32, kW: u32,
    sH: u32, sW: u32,
    pH: u32, pW: u32,
    activation: u32, _pad: u32,
};`

func shaderTiledCNN2BackwardDX(tileSize, kernelVol int) string {
	return fmt.Sprintf(wgslCNN2Bwd2DParamsStruct+`
@group(0) @binding(0) var<uniform>             params:     CNN2Bwd2DParams;
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
    let inHW       = params.inH * params.inW;
    let inVol      = params.inC * inHW;
    if (batchIdx >= params.batchSize) { return; }

    let ic  = inElemFlat / inHW;
    let rem = inElemFlat %% inHW;
    let ih  = rem / params.inW;
    let iw  = rem %% params.inW;

    let oArea = params.outH * params.outW;
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
            for (var kh: u32 = 0u; kh < params.kH; kh++) {
                for (var kw: u32 = 0u; kw < params.kW; kw++) {
                    let vh = i32(ih) + i32(params.pH) - i32(kh);
                    let vw = i32(iw) + i32(params.pW) - i32(kw);
                    if (vh >= 0 && vh %% i32(params.sH) == 0 &&
                        vw >= 0 && vw %% i32(params.sW) == 0) {
                        let oh = u32(vh / i32(params.sH));
                        let ow = u32(vw / i32(params.sW));
                        if (oh < params.outH && ow < params.outW) {
                            let outIdx = (batchIdx * params.filters + f) * oArea + oh * params.outW + ow;
                            let dy = gradOutput[outIdx] * activateDerivative(preAct[outIdx], params.activation);
                            let wCacheIdx = ic * params.kH * params.kW + kh * params.kW + kw;
                            sum += dy * wCache[wCacheIdx];
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

func shaderTiledCNN2BackwardDW(tileSize int) string {
	return fmt.Sprintf(wgslCNN2Bwd2DParamsStruct+`
@group(0) @binding(0) var<uniform>             params:      CNN2Bwd2DParams;
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
    let kVol      = params.inC * params.kH * params.kW;
    if (f >= params.filters) { return; }

    let kHW   = params.kH * params.kW;
    let ic    = kernelPos / kHW;
    let kRem  = kernelPos %% kHW;
    let kh    = kRem / params.kW;
    let kw    = kRem %% params.kW;

    let oArea        = params.outH * params.outW;
    let totalSpatial = params.batchSize * oArea;
    var sum: f32     = 0.0;

    var spatial: u32 = 0u;
    loop {
        if (spatial >= totalSpatial) { break; }

        let loadIdx = spatial + local_id.x;
        if (loadIdx < totalSpatial) {
            let lb      = loadIdx / oArea;
            let loohow  = loadIdx %% oArea;
            let lIdx    = lb * params.filters * oArea + f * oArea + loohow;
            dyCache[local_id.x] = gradOutput[lIdx] * activateDerivative(preAct[lIdx], params.activation);
        } else {
            dyCache[local_id.x] = 0.0;
        }
        workgroupBarrier();

        if (kernelPos < kVol) {
            for (var ti: u32 = 0u; ti < %du; ti++) {
                let bSpatial = spatial + ti;
                if (bSpatial >= totalSpatial) { break; }
                let b     = bSpatial / oArea;
                let oohow = bSpatial %% oArea;
                let oh    = oohow / params.outW;
                let ow    = oohow %% params.outW;

                let ih_i = i32(oh * params.sH + kh) - i32(params.pH);
                let iw_i = i32(ow * params.sW + kw) - i32(params.pW);
                if (ih_i >= 0 && u32(ih_i) < params.inH &&
                    iw_i >= 0 && u32(iw_i) < params.inW) {
                    let inIdx = ((b * params.inC + ic) * params.inH + u32(ih_i)) * params.inW + u32(iw_i);
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

// CNN2Forward runs on-device Conv2d (pre-activation, scale=1). Layout: [batch,inC,inH,inW] → [batch,outC,outH,outW].
func CNN2Forward(input, weights, output []float32, cfg CNN2Config) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu CNN2Forward: %w", initErr)
	}
	nIn := cfg.Batch * cfg.InC * cfg.InH * cfg.InW
	nOut := cfg.Batch * cfg.OutC * cfg.OutH * cfg.OutW
	nW := cfg.OutC * cfg.InC * cfg.KSize * cfg.KSize
	if len(input) < nIn || len(weights) < nW || len(output) < nOut {
		return fmt.Errorf("webgpu CNN2Forward: shape")
	}
	return sess.cnn2Forward(input[:nIn], weights[:nW], output[:nOut], cfg)
}

// CNN2Backward computes gradInput and gradWeights on device (+= into caller-provided zeroed buffers).
func CNN2Backward(gradOut, weights, input, preAct, gradIn, gradW []float32, cfg CNN2Config, act core.ActivationType) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu CNN2Backward: %w", initErr)
	}
	if !CNNTiledBwdOK(act) {
		return fmt.Errorf("webgpu CNN2Backward: unsupported activation %s", act)
	}
	nOut := cfg.Batch * cfg.OutC * cfg.OutH * cfg.OutW
	nIn := cfg.Batch * cfg.InC * cfg.InH * cfg.InW
	nW := cfg.OutC * cfg.InC * cfg.KSize * cfg.KSize
	if len(gradOut) < nOut || len(weights) < nW || len(input) < nIn || len(preAct) < nOut ||
		len(gradIn) < nIn || len(gradW) < nW {
		return fmt.Errorf("webgpu CNN2Backward: shape")
	}
	return sess.cnn2Backward(gradOut[:nOut], weights[:nW], input[:nIn], preAct[:nOut],
		gradIn[:nIn], gradW[:nW], cfg, act)
}

func (s *session) cnn2Forward(input, weights, output []float32, cfg CNN2Config) error {
	k := cfg.KSize
	kernelVol := cfg.InC * k * k
	tileSize := s.cnnTileSize(cfg.MultiCore)
	tiled := s.cnnUseTiledFwd(kernelVol, tileSize)

	var pipe *wgpu.ComputePipeline
	var err error
	if tiled {
		key := cnnPipeKey(tileSize, kernelVol)
		pipe, err = s.getCNNPipe(&s.cnn2FwdPipes, key, shaderTiledCNN2(tileSize, kernelVol),
			fmt.Sprintf("welvet-cnn2-fwd-%d-%d", tileSize, kernelVol))
	} else {
		key := cnnPipeKey(0, 0)
		pipe, err = s.getCNNPipe(&s.cnn2FwdPipes, key, shaderCNN2Scaled, "welvet-cnn2-fwd-scaled")
	}
	if err != nil {
		return err
	}

	dev, q := s.device, s.queue
	inBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-cnn2-in", Contents: wgpu.ToBytes(input),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer inBuf.Destroy()

	wBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-cnn2-w", Contents: wgpu.ToBytes(weights),
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
		Label: "welvet-cnn2-out", Size: outBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer outBuf.Destroy()

	p := cnn2ScaleParams{
		BatchSize: uint32(cfg.Batch),
		InC:       uint32(cfg.InC), InH: uint32(cfg.InH), InW: uint32(cfg.InW),
		OutC:      uint32(cfg.OutC), OutH: uint32(cfg.OutH), OutW: uint32(cfg.OutW),
		KH:        uint32(k), KW: uint32(k),
		SH:        uint32(cfg.Stride), SW: uint32(cfg.Stride),
		PH:        uint32(cfg.Padding), PW: uint32(cfg.Padding),
		Scale:     1,
	}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-cnn2-p", Contents: wgpu.ToBytes([]cnn2ScaleParams{p}),
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

	oArea := cfg.OutH * cfg.OutW
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

func (s *session) cnn2Backward(gradOut, weights, input, preAct, gradIn, gradW []float32, cfg CNN2Config, act core.ActivationType) error {
	k := cfg.KSize
	kernelVol := cfg.InC * k * k
	tileSize := s.cnnTileSize(cfg.MultiCore)
	if !s.cnnUseTiledFwd(kernelVol, tileSize) {
		return fmt.Errorf("webgpu CNN2Backward: kernel too large for tiled shared memory")
	}

	keyDX := cnnPipeKey(tileSize, kernelVol)
	pipeDX, err := s.getCNNPipe(&s.cnn2BwdDXPipes, keyDX, shaderTiledCNN2BackwardDX(tileSize, kernelVol),
		fmt.Sprintf("welvet-cnn2-bwd-dx-%d-%d", tileSize, kernelVol))
	if err != nil {
		return err
	}
	keyDW := cnnDWPipeKey(tileSize)
	pipeDW, err := s.getCNNPipe(&s.cnn2BwdDWPipes, keyDW, shaderTiledCNN2BackwardDW(tileSize),
		fmt.Sprintf("welvet-cnn2-bwd-dw-%d", tileSize))
	if err != nil {
		return err
	}

	dev, q := s.device, s.queue
	bwdP := cnn2Bwd2DParams{
		BatchSize: uint32(cfg.Batch),
		InC:       uint32(cfg.InC), InH: uint32(cfg.InH), InW: uint32(cfg.InW),
		Filters:   uint32(cfg.OutC),
		OutH:      uint32(cfg.OutH), OutW: uint32(cfg.OutW),
		KH:        uint32(k), KW: uint32(k),
		SH:        uint32(cfg.Stride), SW: uint32(cfg.Stride),
		PH:        uint32(cfg.Padding), PW: uint32(cfg.Padding),
		Activation: mapCNNActivation(act),
	}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-cnn2-bwd-p", Contents: wgpu.ToBytes([]cnn2Bwd2DParams{bwdP}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()

	gyBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-cnn2-gy", Contents: wgpu.ToBytes(gradOut),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer gyBuf.Destroy()

	wBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-cnn2-w", Contents: wgpu.ToBytes(weights),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer wBuf.Destroy()

	inBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-cnn2-in", Contents: wgpu.ToBytes(input),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer inBuf.Destroy()

	preBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-cnn2-pre", Contents: wgpu.ToBytes(preAct),
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

	inVol := cfg.InC * cfg.InH * cfg.InW
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
