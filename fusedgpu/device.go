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
	devMu         sync.Mutex
	sharedInst    *wgpu.Instance
	sharedAdapt   *wgpu.Adapter
	sharedDevice  *wgpu.Device
	sharedQueue   *wgpu.Queue
	sharedName    string
	sharedMaxSSBO uint64
	lastProbe     string

	lastHybridDispToken int
	lastHybridUsedSPIRV bool
	lastHybridBackend   string // "wgpu" | "vk" | ""
	lastVKDeviceName    string

	// preferNativeVKPref: nil = platform default (Android ON when built with tag),
	// true/false = explicit user/env preference.
	preferNativeVKPref *bool
)

// LastHybridDispatchesPerToken returns compute dispatches in the last hybrid
// single-token forward, or 0 if none yet.
func LastHybridDispatchesPerToken() int {
	devMu.Lock()
	defer devMu.Unlock()
	return lastHybridDispToken
}

// LastHybridUsedSPIRV reports whether any AOT SPIR-V pipelines were loaded.
func LastHybridUsedSPIRV() bool {
	devMu.Lock()
	defer devMu.Unlock()
	return lastHybridUsedSPIRV
}

// LastHybridBackend returns "vk", "wgpu", or "" for the last hybrid mount/generate path.
func LastHybridBackend() string {
	devMu.Lock()
	defer devMu.Unlock()
	return lastHybridBackend
}

// LastVKDeviceName returns the native Vulkan physical device name when backend=vk.
func LastVKDeviceName() string {
	devMu.Lock()
	defer devMu.Unlock()
	return lastVKDeviceName
}

func noteHybridStats(dispPerToken int, usedSPIRV bool) {
	devMu.Lock()
	defer devMu.Unlock()
	lastHybridDispToken = dispPerToken
	lastHybridUsedSPIRV = usedSPIRV
}

func noteHybridBackend(backend string) {
	devMu.Lock()
	defer devMu.Unlock()
	lastHybridBackend = backend
}

// NoteHybridBackend records which hybrid path was mounted ("vk" or "wgpu").
func NoteHybridBackend(backend string) { noteHybridBackend(backend) }

// NoteVKDeviceName records the native Vulkan device name for Turnip/status checks.
func NoteVKDeviceName(name string) {
	devMu.Lock()
	defer devMu.Unlock()
	lastVKDeviceName = strings.TrimSpace(name)
}

// SetPreferNativeVK sets whether SyncHybridFused should try native Vulkan first.
// On Android builds with birdkit_native_vk, the default is true until overridden.
func SetPreferNativeVK(on bool) {
	devMu.Lock()
	defer devMu.Unlock()
	v := on
	preferNativeVKPref = &v
	_ = os.Setenv("WELVET_NATIVE_VK", map[bool]string{true: "1", false: "0"}[on])
}

// PreferNativeVK reports the effective preference (env WELVET_NATIVE_VK wins if set).
func PreferNativeVK() bool {
	if v := strings.TrimSpace(os.Getenv("WELVET_NATIVE_VK")); v != "" {
		return v == "1" || strings.EqualFold(v, "true") || strings.EqualFold(v, "on")
	}
	devMu.Lock()
	pref := preferNativeVKPref
	devMu.Unlock()
	if pref != nil {
		return *pref
	}
	// Default: prefer native VK on Android when the binary includes it.
	return runtime.GOOS == "android" && NativeVKAvailable()
}

// LastDeviceProbe returns the most recent fusedgpu adapter/limits dump.
func LastDeviceProbe() string {
	devMu.Lock()
	defer devMu.Unlock()
	return lastProbe
}

// MaxStorageBindingLimit returns the current adapter's storage binding limit.
func MaxStorageBindingLimit() uint64 {
	devMu.Lock()
	defer devMu.Unlock()
	return sharedMaxSSBO
}

// resolveBackends picks a native wgpu backend (Metal on macOS/iOS, DX12 on
// Windows, Vulkan elsewhere). Matches welvet/webgpu so gpu_fuse works on ARM Mac.
// Note: birdkit's github.com/openfluke/webgpu InstanceDescriptor only exposes
// Backends (no RequiredFeatures / InstanceFeatureName yet).
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

func formatLimits(l wgpu.Limits) string {
	return fmt.Sprintf(
		"maxBuf=%d maxSSBO=%d maxSSBO/stage=%d maxBindGroups=%d maxComputeWG=%d maxInvoc/WG=%d",
		l.MaxBufferSize,
		l.MaxStorageBufferBindingSize,
		l.MaxStorageBuffersPerShaderStage,
		l.MaxBindGroups,
		l.MaxComputeWorkgroupsPerDimension,
		l.MaxComputeInvocationsPerWorkgroup,
	)
}

func acquireDevice() (inst *wgpu.Instance, adapt *wgpu.Adapter, dev *wgpu.Device, q *wgpu.Queue, name string, err error) {
	devMu.Lock()
	defer devMu.Unlock()
	if sharedDevice != nil {
		return sharedInst, sharedAdapt, sharedDevice, sharedQueue, sharedName, nil
	}

	var report strings.Builder
	fmt.Fprintf(&report, "fusedgpu probe goos=%s goarch=%s WELVET_WGPU_BACKEND=%q\n",
		runtime.GOOS, runtime.GOARCH, strings.TrimSpace(os.Getenv("WELVET_WGPU_BACKEND")))

	inst = wgpu.CreateInstance(resolveBackends())
	if inst == nil {
		msg := "CreateInstance failed"
		report.WriteString(msg + "\n")
		lastProbe = report.String()
		fmt.Print(lastProbe)
		return nil, nil, nil, nil, "", fmt.Errorf(msg)
	}

	opts := []*wgpu.RequestAdapterOptions{
		{PowerPreference: wgpu.PowerPreferenceHighPerformance},
		{PowerPreference: wgpu.PowerPreferenceLowPower},
		{ForceFallbackAdapter: true},
		{},
	}
	labels := []string{"high-perf", "low-power", "force-fallback", "default"}
	for i, opt := range opts {
		a, aerr := inst.RequestAdapter(opt)
		if aerr != nil || a == nil {
			fmt.Fprintf(&report, "  adapter %-14s FAIL: %v\n", labels[i], aerr)
			continue
		}
		info := a.GetInfo()
		lim := a.GetLimits().Limits
		fmt.Fprintf(&report, "  adapter %-14s name=%q vendor=%q arch=%q driver=%q backend=%v type=%v | %s\n",
			labels[i], info.Name, info.VendorName, info.Architecture, info.DriverDescription, info.BackendType, info.AdapterType, formatLimits(lim))
		if adapt == nil {
			adapt = a
			name = info.Name
			fmt.Fprintf(&report, "  → selected %s\n", labels[i])
		} else {
			a.Release()
		}
	}
	if adapt == nil {
		msg := "RequestAdapter: no adapter"
		report.WriteString(msg + "\n")
		lastProbe = report.String()
		fmt.Print(lastProbe)
		inst.Release()
		return nil, nil, nil, nil, "", fmt.Errorf(msg)
	}

	limits := adapt.GetLimits().Limits
	// Cap to fusedgpu design targets without exceeding what the adapter advertises.
	req := limits
	req.MaxStorageBufferBindingSize = minU64(1<<30, limits.MaxStorageBufferBindingSize)
	req.MaxBufferSize = minU64(2<<30, limits.MaxBufferSize)
	if req.MaxStorageBuffersPerShaderStage < 16 {
		req.MaxStorageBuffersPerShaderStage = 16
		if limits.MaxStorageBuffersPerShaderStage > 0 && req.MaxStorageBuffersPerShaderStage > limits.MaxStorageBuffersPerShaderStage {
			req.MaxStorageBuffersPerShaderStage = limits.MaxStorageBuffersPerShaderStage
		}
	}
	fmt.Fprintf(&report, "  try RequestDevice capped: %s\n", formatLimits(req))
	dev, err = adapt.RequestDevice(&wgpu.DeviceDescriptor{
		RequiredLimits: &wgpu.RequiredLimits{Limits: req},
	})
	via := ""
	if err == nil && dev != nil {
		via = "capped"
	} else {
		fmt.Fprintf(&report, "  capped FAIL: %v\n", err)
		fmt.Fprintf(&report, "  try RequestDevice advertised: %s\n", formatLimits(limits))
		dev, err = adapt.RequestDevice(&wgpu.DeviceDescriptor{
			RequiredLimits: &wgpu.RequiredLimits{Limits: limits},
		})
		if err == nil && dev != nil {
			via = "advertised"
		} else {
			fmt.Fprintf(&report, "  advertised FAIL: %v\n", err)
			fmt.Fprintf(&report, "  try RequestDevice(nil)\n")
			dev, err = adapt.RequestDevice(nil)
			if err == nil && dev != nil {
				via = "defaults"
			}
		}
	}
	if err != nil || dev == nil {
		fmt.Fprintf(&report, "  defaults FAIL: %v\n", err)
		fmt.Fprintf(&report, "RESULT: FAIL adapter=%q maxSSBO=%d maxBuf=%d\n",
			name, limits.MaxStorageBufferBindingSize, limits.MaxBufferSize)
		lastProbe = report.String()
		fmt.Print(lastProbe)
		adapt.Release()
		inst.Release()
		return nil, nil, nil, nil, "", fmt.Errorf("RequestDevice: %w", err)
	}
	fmt.Fprintf(&report, "RESULT: OK via=%s adapter=%q maxSSBO=%d maxBuf=%d\n",
		via, name, limits.MaxStorageBufferBindingSize, limits.MaxBufferSize)
	lastProbe = report.String()
	fmt.Print(lastProbe)

	q = dev.GetQueue()
	sharedMaxSSBO = limits.MaxStorageBufferBindingSize
	sharedInst, sharedAdapt, sharedDevice, sharedQueue, sharedName = inst, adapt, dev, q, name
	return inst, adapt, dev, q, name, nil
}

// ReleaseSharedDevice destroys the process-wide fusedgpu wgpu device so Metal/Vulkan
// VRAM from a prior chat fuse can be reclaimed before Hot Potato mounts ASR.
// Next SyncGPU/SyncHybridFused recreates the device.
func ReleaseSharedDevice() {
	devMu.Lock()
	defer devMu.Unlock()
	if sharedQueue != nil {
		sharedQueue.Release()
		sharedQueue = nil
	}
	if sharedDevice != nil {
		sharedDevice.Poll(true, nil)
		sharedDevice.Release()
		sharedDevice = nil
	}
	if sharedAdapt != nil {
		sharedAdapt.Release()
		sharedAdapt = nil
	}
	if sharedInst != nil {
		sharedInst.Release()
		sharedInst = nil
	}
	sharedName = ""
	sharedMaxSSBO = 0
	runtime.GC()
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
	// Keep shared device/adapter/instance for the next SyncGPU.
	e.device, e.queue, e.adapter, e.instance = nil, nil, nil, nil
}
