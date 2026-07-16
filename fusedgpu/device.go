package fusedgpu

import (
	"fmt"
	"sync"

	"github.com/openfluke/webgpu/wgpu"
)

// Shared Vulkan device across sequential gpu_fuse loads so VRAM is not
// fragmented by create/destroy of Instance+Device per format.
var (
	devMu        sync.Mutex
	sharedInst   *wgpu.Instance
	sharedAdapt  *wgpu.Adapter
	sharedDevice *wgpu.Device
	sharedQueue  *wgpu.Queue
	sharedName   string
)

func acquireDevice() (inst *wgpu.Instance, adapt *wgpu.Adapter, dev *wgpu.Device, q *wgpu.Queue, name string, err error) {
	devMu.Lock()
	defer devMu.Unlock()
	if sharedDevice != nil {
		return sharedInst, sharedAdapt, sharedDevice, sharedQueue, sharedName, nil
	}
	inst = wgpu.CreateInstance(&wgpu.InstanceDescriptor{Backends: wgpu.InstanceBackendVulkan})
	if inst == nil {
		return nil, nil, nil, nil, "", fmt.Errorf("CreateInstance failed")
	}
	adapt, err = inst.RequestAdapter(&wgpu.RequestAdapterOptions{
		PowerPreference: wgpu.PowerPreferenceHighPerformance,
	})
	if err != nil || adapt == nil {
		inst.Release()
		return nil, nil, nil, nil, "", fmt.Errorf("RequestAdapter: %w", err)
	}
	info := adapt.GetInfo()
	name = info.Name
	fmt.Printf("Adapter: %s [%v] (shared device)\n", info.Name, info.BackendType)

	limits := adapt.GetLimits().Limits
	limits.MaxStorageBufferBindingSize = minU64(1<<30, limits.MaxStorageBufferBindingSize)
	limits.MaxBufferSize = minU64(2<<30, limits.MaxBufferSize)
	if limits.MaxStorageBuffersPerShaderStage < 12 {
		limits.MaxStorageBuffersPerShaderStage = 12
	}
	dev, err = adapt.RequestDevice(&wgpu.DeviceDescriptor{
		RequiredLimits: &wgpu.RequiredLimits{Limits: limits},
	})
	if err != nil || dev == nil {
		adapt.Release()
		inst.Release()
		return nil, nil, nil, nil, "", fmt.Errorf("RequestDevice: %w", err)
	}
	q = dev.GetQueue()
	sharedInst, sharedAdapt, sharedDevice, sharedQueue, sharedName = inst, adapt, dev, q, name
	return inst, adapt, dev, q, name, nil
}

// releaseModelGPU frees this engine's buffers/pipelines/bind-groups but keeps
// the shared Vulkan device. Destroying the device each SyncGPU leaks ~150–200MB
// host/driver state on this stack and OOMs late in multi-format benches.
func (e *engine) releaseModelGPU() {
	if e == nil {
		return
	}
	if e.device != nil {
		e.device.Poll(true, nil)
	}
	for _, bg := range e.bg {
		if bg != nil {
			bg.Release()
		}
	}
	e.bg = nil
	for _, p := range e.pipe {
		if p != nil {
			p.Release()
		}
	}
	e.pipe = nil
	for _, b := range e.owned {
		if b != nil {
			b.Release()
		}
	}
	e.owned = nil
	e.blocks = nil
	e.embed, e.finalNorm, e.lmScales, e.lmW = nil, nil, nil, nil
	e.step, e.token, e.promptBuf, e.histBuf, e.stagingHist = nil, nil, nil, nil, nil
	e.hidden, e.normed, e.qkvBuf, e.attnOut = nil, nil, nil, nil
	e.inter, e.logits, e.outTok, e.stagingLogits = nil, nil, nil, nil
	e.uGemvQDimH, e.uGemvHInter, e.uGemvVocabH = nil, nil, nil
	e.uQKV, e.uSwiglu, e.uRMS, e.uResidH = nil, nil, nil, nil
	e.uRopeQ, e.uRopeK, e.uAttn, e.uKV = nil, nil, nil, nil
	e.uEmbed, e.uArgMax = nil, nil
	// Keep shared device/adapter/instance for the next SyncGPU.
	e.device, e.queue, e.adapter, e.instance = nil, nil, nil, nil
	e.m = nil
	devMu.Lock()
	if sharedDevice != nil {
		sharedDevice.Poll(true, nil)
	}
	devMu.Unlock()
}
