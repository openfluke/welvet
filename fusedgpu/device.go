package fusedgpu

import (
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync"

	"github.com/openfluke/webgpu/wgpu"
)

// Shared GPU device across sequential gpu_fuse loads so VRAM is not
// fragmented by create/destroy of Instance+Device per format.
var (
	devMu        sync.Mutex
	sharedInst   *wgpu.Instance
	sharedAdapt  *wgpu.Adapter
	sharedDevice *wgpu.Device
	sharedQueue  *wgpu.Queue
	sharedName   string
)

// resolveBackends picks a native wgpu backend (Metal on macOS/iOS, DX12 on
// Windows, Vulkan elsewhere). Matches welvet/webgpu so gpu_fuse works on ARM Mac.
func resolveBackends() *wgpu.InstanceDescriptor {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("WELVET_WGPU_BACKEND"))) {
	case "all":
		return nil
	case "dx12", "d3d12":
		return &wgpu.InstanceDescriptor{Backends: wgpu.InstanceBackendDX12}
	case "vulkan", "vk":
		return &wgpu.InstanceDescriptor{Backends: wgpu.InstanceBackendVulkan}
	case "metal":
		return &wgpu.InstanceDescriptor{Backends: wgpu.InstanceBackendMetal}
	case "gl", "opengl":
		return &wgpu.InstanceDescriptor{Backends: wgpu.InstanceBackendGL}
	}
	switch runtime.GOOS {
	case "darwin", "ios":
		return &wgpu.InstanceDescriptor{Backends: wgpu.InstanceBackendMetal}
	case "windows":
		return &wgpu.InstanceDescriptor{Backends: wgpu.InstanceBackendDX12}
	default:
		return &wgpu.InstanceDescriptor{Backends: wgpu.InstanceBackendVulkan}
	}
}

func acquireDevice() (inst *wgpu.Instance, adapt *wgpu.Adapter, dev *wgpu.Device, q *wgpu.Queue, name string, err error) {
	devMu.Lock()
	defer devMu.Unlock()
	if sharedDevice != nil {
		return sharedInst, sharedAdapt, sharedDevice, sharedQueue, sharedName, nil
	}
	inst = wgpu.CreateInstance(resolveBackends())
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
	if limits.MaxStorageBuffersPerShaderStage < 16 {
		limits.MaxStorageBuffersPerShaderStage = 16
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
// the shared GPU device. Destroying the device each SyncGPU leaks ~150–200MB
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
