package webgpu

import (
	"fmt"
	"strings"

	"github.com/openfluke/webgpu/wgpu"
	"github.com/openfluke/welvet/tiling"
)

// Affine4 g64 sticky weights (MLX AffineQuantized text encoder).
type affGPU struct {
	scales *wgpu.Buffer
	biases *wgpu.Buffer
	words  *wgpu.Buffer
	rows   int
	cols   int
	group  int
	bytes  uint64
}

func (s *session) ensureAffCache() {
	if s.affCache == nil {
		s.affCache = make(map[uintptr]*affGPU)
	}
}

// ClearAffineWeightCache releases sticky AffinePacked GPU weights.
func ClearAffineWeightCache() {
	mu.Lock()
	defer mu.Unlock()
	if sess == nil {
		return
	}
	sess.clearAffCacheLocked()
}

func (s *session) clearAffCacheLocked() {
	if s == nil {
		return
	}
	for _, e := range s.affCache {
		if e == nil {
			continue
		}
		if e.scales != nil {
			e.scales.Destroy()
		}
		if e.biases != nil {
			e.biases.Destroy()
		}
		if e.words != nil {
			e.words.Destroy()
		}
	}
	s.affCache = nil
	s.affCacheBytes = 0
	s.affCacheFull = false
}

// AffineWeightCacheBytes reports resident AffinePacked VRAM (approx).
func AffineWeightCacheBytes() uint64 {
	mu.Lock()
	defer mu.Unlock()
	if sess == nil {
		return 0
	}
	return sess.affCacheBytes
}

var errAffineVRAMFull = fmt.Errorf("webgpu: affine weight VRAM budget full")

// IsAffineVRAMFull reports whether further Affine uploads should stop.
func IsAffineVRAMFull(err error) bool {
	if err == nil {
		return false
	}
	if err == errAffineVRAMFull {
		return true
	}
	s := strings.ToLower(err.Error())
	return strings.Contains(s, "vram") || strings.Contains(s, "budget full") || strings.Contains(s, "not enough memory")
}

// WarmAffineWeight uploads scales+biases+words once (Affine4, group typically 64).
func WarmAffineWeight(key uintptr, scales, biases []float32, words []uint32, rows, cols, group int) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return initErr
	}
	if group <= 0 {
		group = 64
	}
	if cols%group != 0 || cols%8 != 0 {
		return fmt.Errorf("webgpu WarmAffineWeight: cols")
	}
	needWords := (rows * cols) / 8
	needSB := rows * (cols / group)
	if len(words) < needWords || len(scales) < needSB || len(biases) < needSB {
		return fmt.Errorf("webgpu WarmAffineWeight: short buffers")
	}
	mu.Lock()
	defer mu.Unlock()
	_, err := sess.uploadAffineLocked(key, scales[:needSB], biases[:needSB], words[:needWords], rows, cols, group)
	return err
}

func (s *session) uploadAffineLocked(key uintptr, scales, biases []float32, words []uint32, rows, cols, group int) (*affGPU, error) {
	s.ensureAffCache()
	if e, ok := s.affCache[key]; ok && e != nil {
		return e, nil
	}
	if s.affCacheFull {
		return nil, errAffineVRAMFull
	}
	nbytes := uint64(len(scales)*4 + len(biases)*4 + len(words)*4)
	const softCap = 2800 << 20
	if s.affCacheBytes+nbytes > softCap {
		s.affCacheFull = true
		return nil, errAffineVRAMFull
	}
	dev := s.device
	scBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-aff-sc", Contents: wgpu.ToBytes(scales),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, err
	}
	bBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-aff-b", Contents: wgpu.ToBytes(biases),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		scBuf.Destroy()
		return nil, err
	}
	wBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-aff-w", Contents: wgpu.ToBytes(words),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		scBuf.Destroy()
		bBuf.Destroy()
		return nil, err
	}
	e := &affGPU{scales: scBuf, biases: bBuf, words: wBuf, rows: rows, cols: cols, group: group, bytes: nbytes}
	s.affCache[key] = e
	s.affCacheBytes += nbytes
	return e, nil
}

// HasAffineWeight reports whether key is resident.
func HasAffineWeight(key uintptr) bool {
	ensure()
	mu.Lock()
	defer mu.Unlock()
	if sess == nil || sess.affCache == nil {
		return false
	}
	_, ok := sess.affCache[key]
	return ok
}

// DenseGEMVAffineResident runs Affine4 GEMV with sticky weights.
func DenseGEMVAffineResident(key uintptr, scales, biases []float32, words []uint32, x, y []float32, batch, in, out, group int) error {
	ensure()
	if !haveGPU || sess == nil {
		if initErr == nil {
			initErr = fmt.Errorf("webgpu: no device")
		}
		return fmt.Errorf("webgpu DenseGEMVAffineResident: %w", initErr)
	}
	if key == 0 || in <= 0 || in%8 != 0 || len(x) < batch*in || len(y) < batch*out {
		return fmt.Errorf("webgpu DenseGEMVAffineResident: shape")
	}
	if group <= 0 {
		group = 64
	}
	mu.Lock()
	sess.ensureAffCache()
	if e, ok := sess.affCache[key]; ok && e != nil {
		mu.Unlock()
		return sess.gemvAffineResident(e, x, y, batch, in, out)
	}
	mu.Unlock()

	needWords := (out * in) / 8
	needSB := out * (in / group)
	if len(words) < needWords || len(scales) < needSB || len(biases) < needSB {
		return fmt.Errorf("webgpu DenseGEMVAffineResident: short")
	}
	mu.Lock()
	e, err := sess.uploadAffineLocked(key, scales[:needSB], biases[:needSB], words[:needWords], out, in, group)
	mu.Unlock()
	if err != nil {
		return err
	}
	return sess.gemvAffineResident(e, x, y, batch, in, out)
}

func (s *session) ensureAffinePipe() error {
	if s.pipeAffine != nil {
		return nil
	}
	p, err := makePipeline(s.device, ShaderDenseAffine4, "welvet-dense-affine4")
	if err != nil {
		return err
	}
	s.pipeAffine = p
	return nil
}

func (s *session) gemvAffineResident(e *affGPU, x, y []float32, batch, in, out int) error {
	if e == nil {
		return fmt.Errorf("webgpu: nil affine weights")
	}
	if err := s.ensureAffinePipe(); err != nil {
		return err
	}
	return s.dispatchAffine(s.device, s.queue, s.pipeAffine, e, x, y, batch, in, out)
}

type gpuParamsAffine struct {
	Batch      uint32
	InputSize  uint32
	OutputSize uint32
	GroupSize  uint32
}

func (s *session) dispatchAffine(dev *wgpu.Device, q *wgpu.Queue, pipe *wgpu.ComputePipeline,
	e *affGPU, x, y []float32, batch, in, out int) error {

	const wg = 64
	xBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-aff-x", Contents: wgpu.ToBytes(x[:batch*in]),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer xBuf.Destroy()

	yBytes := uint64(batch * out * 4)
	if yBytes < 64 {
		yBytes = 64
	}
	yBuf, err := dev.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "welvet-aff-y", Size: yBytes,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer yBuf.Destroy()

	p := gpuParamsAffine{
		Batch: uint32(batch), InputSize: uint32(in), OutputSize: uint32(out), GroupSize: uint32(e.group),
	}
	pBuf, err := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "welvet-aff-p", Contents: wgpu.ToBytes([]gpuParamsAffine{p}),
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
			{Binding: 1, Buffer: xBuf, Offset: 0, Size: xBuf.GetSize()},
			{Binding: 2, Buffer: e.scales, Offset: 0, Size: e.scales.GetSize()},
			{Binding: 3, Buffer: e.biases, Offset: 0, Size: e.biases.GetSize()},
			{Binding: 4, Buffer: e.words, Offset: 0, Size: e.words.GetSize()},
			{Binding: 5, Buffer: yBuf, Offset: 0, Size: yBuf.GetSize()},
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
	pass.DispatchWorkgroups(tiling.GPUWorkgroupsX(out, wg), uint32(batch), 1)
	pass.End()
	cmd, err := enc.Finish(nil)
	if err != nil {
		return err
	}
	q.Submit(cmd)

	outY, err := readbackF32(dev, q, yBuf, batch*out)
	if err != nil {
		return err
	}
	copy(y, outY)
	return nil
}

// ShaderDenseAffine4 — MLX AffineQuantized 4-bit g64: w = s*code + β, code∈[0,15], 8 codes/u32 LSB-first.
const ShaderDenseAffine4 = `
struct Params { batch: u32, inputSize: u32, outputSize: u32, groupSize: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> biases: array<f32>;
@group(0) @binding(4) var<storage, read> weights: array<u32>;
@group(0) @binding(5) var<storage, read_write> Y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    let b = gid.y;
    if (o >= params.outputSize || b >= params.batch) { return; }
    let group = params.groupSize;
    let groups = params.inputSize / group;
    let wordsPerRow = params.inputSize / 8u;
    let wordsInGroup = group / 8u;
    let xBase = b * params.inputSize;
    var acc: f32 = 0.0;
    for (var g: u32 = 0u; g < groups; g++) {
        let sb = o * groups + g;
        let s = scales[sb];
        let beta = biases[sb];
        let colBase = g * group;
        let baseWord = o * wordsPerRow + g * wordsInGroup;
        var codeDot: f32 = 0.0;
        var sumX: f32 = 0.0;
        for (var wi: u32 = 0u; wi < wordsInGroup; wi++) {
            let packed = weights[baseWord + wi];
            let xb = xBase + colBase + wi * 8u;
            for (var n: u32 = 0u; n < 8u; n++) {
                let code = f32((packed >> (n * 4u)) & 0xFu);
                let xv = X[xb + n];
                codeDot += code * xv;
                sumX += xv;
            }
        }
        acc += s * codeDot + beta * sumX;
    }
    Y[b * params.outputSize + o] = acc;
}
`
