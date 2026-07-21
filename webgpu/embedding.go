package webgpu

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
	"github.com/openfluke/welvet/tiling"
)

type embeddingParams struct {
	VocabSize  uint32
	HiddenSize uint32
	NumTokens  uint32
}

// EmbeddingGather loads rows from weights by token indices on a real WebGPU device.
func EmbeddingGather(indices []uint32, weights, output []float32, vocab, hidden, nTok int) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu EmbeddingGather: %w", initErr)
	}
	if vocab <= 0 || hidden <= 0 || nTok <= 0 || len(indices) < nTok ||
		len(weights) < vocab*hidden || len(output) < nTok*hidden {
		return fmt.Errorf("webgpu EmbeddingGather: shape")
	}
	if err := sess.ensureEmbeddingPipes(); err != nil {
		return err
	}
	return sess.embeddingGather(indices, weights, output, vocab, hidden, nTok)
}

// EmbeddingScatter accumulates gradOutput into gradWeights by token indices on device.
// gradWeights must be zero-initialized; the kernel uses += (same as loom).
func EmbeddingScatter(indices []uint32, gradOutput, gradWeights []float32, vocab, hidden, nTok int) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu EmbeddingScatter: %w", initErr)
	}
	if vocab <= 0 || hidden <= 0 || nTok <= 0 || len(indices) < nTok ||
		len(gradOutput) < nTok*hidden || len(gradWeights) < vocab*hidden {
		return fmt.Errorf("webgpu EmbeddingScatter: shape")
	}
	if err := sess.ensureEmbeddingPipes(); err != nil {
		return err
	}
	return sess.embeddingScatter(indices, gradOutput, gradWeights, vocab, hidden, nTok)
}

func (s *session) ensureEmbeddingPipes() error {
	if s.pipeEmbedding != nil {
		return nil
	}
	var err error
	s.pipeEmbedding, err = makePipeline(s.device, ShaderEmbedding, "welvet-embedding")
	if err != nil {
		return err
	}
	s.pipeEmbeddingBwd, err = makePipeline(s.device, ShaderEmbeddingBackward, "welvet-embedding-bwd")
	return err
}

func (s *session) embeddingGather(indices []uint32, weights, output []float32, vocab, hidden, nTok int) error {
	const wg = 64
	dev, q := s.device, s.queue
	nElem := nTok * hidden

	idxBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-emb-idx", Contents: wgpu.ToBytes(indices[:nTok]),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer idxBuf.Destroy()

	wBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-emb-w", Contents: wgpu.ToBytes(weights[:vocab*hidden]),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer wBuf.Destroy()

	outBytes := uint64(nElem * 4)
	if outBytes < 64 {
		outBytes = 64
	}
	outBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "welvet-emb-out", Size: outBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer outBuf.Destroy()

	p := embeddingParams{
		VocabSize: uint32(vocab), HiddenSize: uint32(hidden), NumTokens: uint32(nTok),
	}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-emb-p", Contents: wgpu.ToBytes([]embeddingParams{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()

	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.pipeEmbedding.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: idxBuf, Offset: 0, Size: idxBuf.GetSize()},
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
	pass.SetPipeline(s.pipeEmbedding)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(tiling.GPUWorkgroupsX(nElem, wg), 1, 1)
	pass.End()
	cmd, err := enc.Finish(nil)
	if err != nil {
		return err
	}
	q.Submit(cmd)

	outData, err := readbackF32(dev, q, outBuf, nElem)
	if err != nil {
		return err
	}
	copy(output, outData)
	return nil
}

func (s *session) embeddingScatter(indices []uint32, gradOut, gradW []float32, vocab, hidden, nTok int) error {
	const wg = 64
	dev, q := s.device, s.queue
	nElem := nTok * hidden
	nW := vocab * hidden

	idxBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-emb-bwd-idx", Contents: wgpu.ToBytes(indices[:nTok]),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer idxBuf.Destroy()

	goBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-emb-bwd-go", Contents: wgpu.ToBytes(gradOut[:nElem]),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer goBuf.Destroy()

	dwBytes := uint64(nW * 4)
	if dwBytes < 64 {
		dwBytes = 64
	}
	dwBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-emb-bwd-dw", Contents: wgpu.ToBytes(make([]float32, nW)),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer dwBuf.Destroy()

	p := embeddingParams{
		VocabSize: uint32(vocab), HiddenSize: uint32(hidden), NumTokens: uint32(nTok),
	}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-emb-bwd-p", Contents: wgpu.ToBytes([]embeddingParams{p}),
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer pBuf.Destroy()

	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: s.pipeEmbeddingBwd.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: pBuf, Offset: 0, Size: pBuf.GetSize()},
			{Binding: 1, Buffer: idxBuf, Offset: 0, Size: idxBuf.GetSize()},
			{Binding: 2, Buffer: goBuf, Offset: 0, Size: goBuf.GetSize()},
			{Binding: 3, Buffer: dwBuf, Offset: 0, Size: dwBuf.GetSize()},
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
	pass.SetPipeline(s.pipeEmbeddingBwd)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(tiling.GPUWorkgroupsX(nElem, wg), 1, 1)
	pass.End()
	cmd, err := enc.Finish(nil)
	if err != nil {
		return err
	}
	q.Submit(cmd)

	outW, err := readbackF32(dev, q, dwBuf, nW)
	if err != nil {
		return err
	}
	copy(gradW, outW)
	return nil
}

// ShaderEmbedding — gather rows from the embedding table by token index.
const ShaderEmbedding = `
struct Params {
    vocabSize: u32,
    hiddenSize: u32,
    numTokens: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    if (tid >= params.numTokens * params.hiddenSize) { return; }

    let tokenIdx = tid / params.hiddenSize;
    let dimIdx = tid % params.hiddenSize;
    let vocabIdx = indices[tokenIdx];

    if (vocabIdx >= params.vocabSize) {
        output[tid] = 0.0;
    } else {
        output[tid] = weights[vocabIdx * params.hiddenSize + dimIdx];
    }
}
`

// ShaderEmbeddingBackward — scatter gradOutput into gradWeights (+= per token row).
const ShaderEmbeddingBackward = `
struct Params {
    vocabSize: u32,
    hiddenSize: u32,
    numTokens: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> gradOutput: array<f32>;
@group(0) @binding(3) var<storage, read_write> gradWeights: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    if (tid >= params.numTokens * params.hiddenSize) { return; }

    let tokenIdx = tid / params.hiddenSize;
    let dimIdx = tid % params.hiddenSize;
    let vocabIdx = indices[tokenIdx];

    if (vocabIdx < params.vocabSize) {
        gradWeights[vocabIdx * params.hiddenSize + dimIdx] += gradOutput[tid];
    }
}
`
