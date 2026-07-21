package webgpu

import (
	"fmt"

	"github.com/openfluke/webgpu/wgpu"
	"github.com/openfluke/welvet/core"
)

const cnnWGSLActivateDeriv = `
fn activateDerivative(v: f32, act: u32) -> f32 {
    if (act == 0u) { if (v <= 0.0) { return 0.0; } return 1.0; }
    if (act == 1u) { let sig = 1.0 / (1.0 + exp(-v)); return sig * (1.0 + v * (1.0 - sig)); }
    if (act == 3u) { let t = tanh(v); return 1.0 - t * t; }
    if (act == 4u) { let s = 1.0 / (1.0 + exp(-v)); return s * (1.0 - s); }
    if (act == 5u) { if (v > 0.0) { return 1.0; } return 0.01; }
    if (act == 6u) { if (v > 0.0) { return 2.0 * v; } return 0.0; }
    return 1.0;
}`

func mapCNNActivation(act core.ActivationType) uint32 {
	switch act {
	case core.ActivationReLU:
		return 0
	case core.ActivationSilu:
		return 1
	case core.ActivationTanh:
		return 3
	case core.ActivationSigmoid:
		return 4
	case core.ActivationLeakyReLU:
		return 5
	case core.ActivationReLU2:
		return 6
	default:
		return 99
	}
}

// CNNTiledBwdOK reports whether backward tiled shaders support this activation.
func CNNTiledBwdOK(act core.ActivationType) bool {
	switch act {
	case core.ActivationLinear, core.ActivationReLU, core.ActivationSilu,
		core.ActivationTanh, core.ActivationSigmoid,
		core.ActivationLeakyReLU, core.ActivationReLU2:
		return true
	default:
		return false
	}
}

func (s *session) ensureCNNLimits() {
	if s.wgStorageMax != 0 {
		return
	}
	limits := s.device.GetLimits().Limits
	s.wgStorageMax = limits.MaxComputeWorkgroupStorageSize
	if s.wgStorageMax == 0 {
		s.wgStorageMax = 16384
	}
	s.maxInvPerWG = limits.MaxComputeInvocationsPerWorkgroup
	if s.maxInvPerWG == 0 {
		s.maxInvPerWG = 256
	}
}

func (s *session) cnnTileSize(multiCore bool) int {
	s.ensureCNNLimits()
	mc := int(s.maxInvPerWG)
	if mc <= 0 || mc > 256 {
		mc = 256
	}
	mc = (mc / 64) * 64
	if mc < 64 {
		mc = 64
	}
	sc := 64
	if multiCore {
		return mc
	}
	return sc
}

func (s *session) cnnUseTiledFwd(kernelVol int, tileSize int) bool {
	s.ensureCNNLimits()
	need := uint32(kernelVol * 4)
	if need > s.wgStorageMax {
		return false
	}
	if uint32(tileSize) > s.maxInvPerWG {
		return false
	}
	return true
}

func cnnPipeKey(tileSize, kernelVol int) uint64 {
	return uint64(tileSize)<<32 | uint64(uint32(kernelVol))
}

func cnnDWPipeKey(tileSize int) uint64 {
	return uint64(tileSize)
}

func (s *session) getCNNPipe(cache *map[uint64]*wgpu.ComputePipeline, key uint64, code, label string) (*wgpu.ComputePipeline, error) {
	if *cache == nil {
		*cache = make(map[uint64]*wgpu.ComputePipeline)
	}
	if p, ok := (*cache)[key]; ok {
		return p, nil
	}
	p, err := makePipeline(s.device, code, label)
	if err != nil {
		return nil, fmt.Errorf("webgpu %s: %w", label, err)
	}
	(*cache)[key] = p
	return p, nil
}

func zeroF32Buffer(dev *wgpu.Device, n int) (*wgpu.Buffer, error) {
	size := uint64(n * 4)
	if size < 64 {
		size = 64
	}
	zeros := make([]float32, (size+3)/4)
	return dev.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    "welvet-cnn-zero",
		Contents: wgpu.ToBytes(zeros),
		Usage:    wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc,
	})
}
