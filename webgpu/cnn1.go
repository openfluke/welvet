package webgpu

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
	"github.com/openfluke/welvet/core"
)

// CNN1Config holds tiled Conv1d dispatch geometry (loom [batch, channels, length] layout).
type CNN1Config struct {
	Batch, InC, InL, OutC, OutL, KSize, Stride, Padding int
	MultiCore                                           bool
}

const shaderCNN1Scaled = `
struct CNN1ScaleParams {
    batchSize: u32,
    inC: u32, inL: u32,
    outC: u32, outL: u32,
    kSize: u32, stride: u32, padding: u32,
    scale: f32,
    _p1: u32, _p2: u32, _p3: u32, _p4: u32, _p5: u32, _p6: u32, _p7: u32,
};

@group(0) @binding(0) var<uniform>             params:  CNN1ScaleParams;
@group(0) @binding(1) var<storage, read>       input:   array<f32>;
@group(0) @binding(2) var<storage, read>       weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output:  array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let outPos    = global_id.x;
    let filterIdx = global_id.y;
    let batchIdx  = global_id.z;
    if (outPos >= params.outL || filterIdx >= params.outC || batchIdx >= params.batchSize) { return; }

    var sum: f32 = 0.0;
    for (var ic: u32 = 0u; ic < params.inC; ic++) {
        for (var k: u32 = 0u; k < params.kSize; k++) {
            let inPos = i32(outPos * params.stride + k) - i32(params.padding);
            if (inPos >= 0 && u32(inPos) < params.inL) {
                let inIdx  = batchIdx * params.inC * params.inL + ic * params.inL + u32(inPos);
                let wIdx   = filterIdx * params.inC * params.kSize + ic * params.kSize + k;
                sum += input[inIdx] * weights[wIdx];
            }
        }
    }
    output[batchIdx * params.outC * params.outL + filterIdx * params.outL + outPos] = sum * params.scale;
}
`

func shaderTiledCNN1(tileSize, kernelVol int) string {
	return fmt.Sprintf(`
struct CNN1ScaleParams {
    batchSize: u32,
    inC: u32, inL: u32,
    outC: u32, outL: u32,
    kSize: u32, stride: u32, padding: u32,
    scale: f32,
    _p1: u32, _p2: u32, _p3: u32, _p4: u32, _p5: u32, _p6: u32, _p7: u32,
};

@group(0) @binding(0) var<uniform>             params:  CNN1ScaleParams;
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
    let kVol: u32 = %du;
    let wBase     = filterIdx * kVol;

    var i: u32 = local_id.x;
    loop {
        if (i >= kVol) { break; }
        wCache[i] = weights[wBase + i];
        i += %du;
    }
    workgroupBarrier();

    let outPos = global_id.x;
    if (outPos >= params.outL || filterIdx >= params.outC || batchIdx >= params.batchSize) { return; }

    var sum: f32 = 0.0;
    for (var ic: u32 = 0u; ic < params.inC; ic++) {
        for (var k: u32 = 0u; k < params.kSize; k++) {
            let inPos = i32(outPos * params.stride + k) - i32(params.padding);
            if (inPos >= 0 && u32(inPos) < params.inL) {
                let inIdx    = batchIdx * params.inC * params.inL + ic * params.inL + u32(inPos);
                let cacheIdx = ic * params.kSize + k;
                sum += input[inIdx] * wCache[cacheIdx];
            }
        }
    }
    output[batchIdx * params.outC * params.outL + filterIdx * params.outL + outPos] = sum * params.scale;
}
`, kernelVol, tileSize, kernelVol, tileSize)
}

type cnn1ScaleParams struct {
	BatchSize uint32
	InC       uint32
	InL       uint32
	OutC      uint32
	OutL      uint32
	KSize     uint32
	Stride    uint32
	Padding   uint32
	Scale     float32
	Pad1      uint32
	Pad2      uint32
	Pad3      uint32
	Pad4      uint32
	Pad5      uint32
	Pad6      uint32
	Pad7      uint32
}

type cnn1Bwd1DParams struct {
	BatchSize  uint32
	InC        uint32
	InL        uint32
	Filters    uint32
	OutL       uint32
	KSize      uint32
	Stride     uint32
	Padding    uint32
	Activation uint32
	Pad1       uint32
	Pad2       uint32
	Pad3       uint32
	Pad4       uint32
	Pad5       uint32
	Pad6       uint32
	Pad7       uint32
}

const wgslCNN1Bwd1DParamsStruct = `
struct CNN1Bwd1DParams {
    batchSize: u32,
    inC: u32, inL: u32,
    filters: u32, outL: u32,
    kSize: u32, stride: u32, padding: u32,
    activation: u32,
    _p1: u32, _p2: u32, _p3: u32, _p4: u32, _p5: u32, _p6: u32, _p7: u32,
};`

func shaderTiledCNN1BackwardDX(tileSize, kernelVol int) string {
	return fmt.Sprintf(wgslCNN1Bwd1DParamsStruct+`
@group(0) @binding(0) var<uniform>             params:     CNN1Bwd1DParams;
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
    let inVol      = params.inC * params.inL;
    if (batchIdx >= params.batchSize) { return; }

    let ic    = inElemFlat / params.inL;
    let inPos = inElemFlat %% params.inL;

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
            for (var k: u32 = 0u; k < params.kSize; k++) {
                let v = i32(inPos) + i32(params.padding) - i32(k);
                if (v >= 0 && v %% i32(params.stride) == 0) {
                    let outPos = u32(v / i32(params.stride));
                    if (outPos < params.outL) {
                        let outIdx = (batchIdx * params.filters + f) * params.outL + outPos;
                        let dy = gradOutput[outIdx] * activateDerivative(preAct[outIdx], params.activation);
                        let wCacheIdx = ic * params.kSize + k;
                        sum += dy * wCache[wCacheIdx];
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

func shaderTiledCNN1BackwardDW(tileSize int) string {
	return fmt.Sprintf(wgslCNN1Bwd1DParamsStruct+`
@group(0) @binding(0) var<uniform>             params:      CNN1Bwd1DParams;
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
    let kernelPos  = global_id.x;
    let f          = global_id.y;
    let kVol       = params.inC * params.kSize;
    if (f >= params.filters) { return; }

    let ic = kernelPos / params.kSize;
    let k  = kernelPos %% params.kSize;

    let totalSpatial = params.batchSize * params.outL;
    var sum: f32     = 0.0;

    var spatial: u32 = 0u;
    loop {
        if (spatial >= totalSpatial) { break; }

        let loadIdx = spatial + local_id.x;
        if (loadIdx < totalSpatial) {
            let lb     = loadIdx / params.outL;
            let lOutPos = loadIdx %% params.outL;
            let lIdx   = lb * params.filters * params.outL + f * params.outL + lOutPos;
            dyCache[local_id.x] = gradOutput[lIdx] * activateDerivative(preAct[lIdx], params.activation);
        } else {
            dyCache[local_id.x] = 0.0;
        }
        workgroupBarrier();

        if (kernelPos < kVol) {
            for (var ti: u32 = 0u; ti < %du; ti++) {
                let bSpatial = spatial + ti;
                if (bSpatial >= totalSpatial) { break; }
                let b      = bSpatial / params.outL;
                let outPos = bSpatial %% params.outL;

                let inPos_i = i32(outPos * params.stride + k) - i32(params.padding);
                if (inPos_i >= 0 && u32(inPos_i) < params.inL) {
                    let inIdx = (b * params.inC + ic) * params.inL + u32(inPos_i);
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

// CNN1Forward runs on-device Conv1d (pre-activation, scale=1). Layout: [batch,inC,inL] → [batch,outC,outL].
func CNN1Forward(input, weights, output []float32, cfg CNN1Config) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu CNN1Forward: %w", initErr)
	}
	nIn := cfg.Batch * cfg.InC * cfg.InL
	nOut := cfg.Batch * cfg.OutC * cfg.OutL
	nW := cfg.OutC * cfg.InC * cfg.KSize
	if len(input) < nIn || len(weights) < nW || len(output) < nOut {
		return fmt.Errorf("webgpu CNN1Forward: shape")
	}
	return sess.cnn1Forward(input[:nIn], weights[:nW], output[:nOut], cfg)
}

// CNN1Backward computes gradInput and gradWeights on device (+= into caller-provided zeroed buffers).
func CNN1Backward(gradOut, weights, input, preAct, gradIn, gradW []float32, cfg CNN1Config, act core.ActivationType) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu CNN1Backward: %w", initErr)
	}
	if !CNNTiledBwdOK(act) {
		return fmt.Errorf("webgpu CNN1Backward: unsupported activation %s", act)
	}
	nOut := cfg.Batch * cfg.OutC * cfg.OutL
	nIn := cfg.Batch * cfg.InC * cfg.InL
	nW := cfg.OutC * cfg.InC * cfg.KSize
	if len(gradOut) < nOut || len(weights) < nW || len(input) < nIn || len(preAct) < nOut ||
		len(gradIn) < nIn || len(gradW) < nW {
		return fmt.Errorf("webgpu CNN1Backward: shape")
	}
	return sess.cnn1Backward(gradOut[:nOut], weights[:nW], input[:nIn], preAct[:nOut],
		gradIn[:nIn], gradW[:nW], cfg, act)
}

func (s *session) cnn1Forward(input, weights, output []float32, cfg CNN1Config) error {
	kernelVol := cfg.InC * cfg.KSize
	tileSize := s.cnnTileSize(cfg.MultiCore)
	tiled := s.cnnUseTiledFwd(kernelVol, tileSize)

	var pipe *wgpu.ComputePipeline
	var err error
	if tiled {
		key := cnnPipeKey(tileSize, kernelVol)
		pipe, err = s.getCNNPipe(&s.cnn1FwdPipes, key, shaderTiledCNN1(tileSize, kernelVol),
			fmt.Sprintf("welvet-cnn1-fwd-%d-%d", tileSize, kernelVol))
	} else {
		key := cnnPipeKey(0, 0)
		pipe, err = s.getCNNPipe(&s.cnn1FwdPipes, key, shaderCNN1Scaled, "welvet-cnn1-fwd-scaled")
	}
	if err != nil {
		return err
	}

	dev, q := s.device, s.queue
	inBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-cnn1-in", Contents: wgpu.ToBytes(input),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer inBuf.Destroy()

	wBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-cnn1-w", Contents: wgpu.ToBytes(weights),
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
		Label: "welvet-cnn1-out", Size: outBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer outBuf.Destroy()

	p := cnn1ScaleParams{
		BatchSize: uint32(cfg.Batch),
		InC:       uint32(cfg.InC),
		InL:       uint32(cfg.InL),
		OutC:      uint32(cfg.OutC),
		OutL:      uint32(cfg.OutL),
		KSize:     uint32(cfg.KSize),
		Stride:    uint32(cfg.Stride),
		Padding:   uint32(cfg.Padding),
		Scale:     1,
	}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-cnn1-p", Contents: wgpu.ToBytes([]cnn1ScaleParams{p}),
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

	enc, err := dev.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipe)
	pass.SetBindGroup(0, bg, nil)
	if tiled {
		pass.DispatchWorkgroups(
			(uint32(cfg.OutL)+uint32(tileSize)-1)/uint32(tileSize),
			uint32(cfg.OutC),
			uint32(cfg.Batch),
		)
	} else {
		pass.DispatchWorkgroups(
			(uint32(cfg.OutL)+63)/64,
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

func (s *session) cnn1Backward(gradOut, weights, input, preAct, gradIn, gradW []float32, cfg CNN1Config, act core.ActivationType) error {
	kernelVol := cfg.InC * cfg.KSize
	tileSize := s.cnnTileSize(cfg.MultiCore)
	if !s.cnnUseTiledFwd(kernelVol, tileSize) {
		return fmt.Errorf("webgpu CNN1Backward: kernel too large for tiled shared memory")
	}

	keyDX := cnnPipeKey(tileSize, kernelVol)
	pipeDX, err := s.getCNNPipe(&s.cnn1BwdDXPipes, keyDX, shaderTiledCNN1BackwardDX(tileSize, kernelVol),
		fmt.Sprintf("welvet-cnn1-bwd-dx-%d-%d", tileSize, kernelVol))
	if err != nil {
		return err
	}
	keyDW := cnnDWPipeKey(tileSize)
	pipeDW, err := s.getCNNPipe(&s.cnn1BwdDWPipes, keyDW, shaderTiledCNN1BackwardDW(tileSize),
		fmt.Sprintf("welvet-cnn1-bwd-dw-%d", tileSize))
	if err != nil {
		return err
	}

	dev, q := s.device, s.queue
	bwdP := cnn1Bwd1DParams{
		BatchSize:  uint32(cfg.Batch),
		InC:        uint32(cfg.InC),
		InL:        uint32(cfg.InL),
		Filters:    uint32(cfg.OutC),
		OutL:       uint32(cfg.OutL),
		KSize:      uint32(cfg.KSize),
		Stride:     uint32(cfg.Stride),
		Padding:    uint32(cfg.Padding),
		Activation: mapCNNActivation(act),
	}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-cnn1-bwd-p", Contents: wgpu.ToBytes([]cnn1Bwd1DParams{bwdP}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()

	gyBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-cnn1-gy", Contents: wgpu.ToBytes(gradOut),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer gyBuf.Destroy()

	wBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-cnn1-w", Contents: wgpu.ToBytes(weights),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer wBuf.Destroy()

	inBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-cnn1-in", Contents: wgpu.ToBytes(input),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer inBuf.Destroy()

	preBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-cnn1-pre", Contents: wgpu.ToBytes(preAct),
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

	enc, err := dev.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pipeDX)
	pass.SetBindGroup(0, bgDX, nil)
	pass.DispatchWorkgroups(
		(uint32(cfg.InC*cfg.InL)+uint32(tileSize)-1)/uint32(tileSize),
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
