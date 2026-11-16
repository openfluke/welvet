// Package accel enables optional native backends for JSON/Exec.Backend placement.
//
// Use Ensure before placing layers with BackendCuda / Metal / Rocm / Vulkan so
// EnableRouting owns the process-wide dispatcher. Probe reports what this host
// can attempt. Close releases routers (optional; process exit is fine).
package accel

import (
	"fmt"
	"runtime"
	"strings"
	"sync"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/nativecuda"
	"github.com/openfluke/welvet/nativemetal"
	"github.com/openfluke/welvet/nativemlx"
	"github.com/openfluke/welvet/nativeqnn"
	"github.com/openfluke/welvet/nativerocm"
	"github.com/openfluke/welvet/nativevulkan"

	_ "github.com/openfluke/welvet/nativecpp" // PreferNativeCPU / NativeCPUAvailable
)

// Info describes one backend for Probe / CABI list_backends.
type Info struct {
	ID        string `json:"id"`
	Backend   int    `json:"backend"`
	Available bool   `json:"available"`
	Routing   string `json:"routing,omitempty"` // "enable" | "sync" | ""
	Note      string `json:"note,omitempty"`
}

type closer interface{ Close() error }

var (
	mu       sync.Mutex
	routers  = map[core.Backend]closer{}
	lastNote sync.Map // backend → last ensure note
)

// Parse maps a CABI / Python backend string onto core.Backend.
func Parse(name string) (core.Backend, error) {
	s := strings.ToLower(strings.TrimSpace(name))
	s = strings.ReplaceAll(s, "-", "_")
	switch s {
	case "", "cpu", "cpu_tiled", "cputiled":
		return core.BackendCPUTiled, nil
	case "simd":
		return core.BackendSIMD, nil
	case "webgpu", "wgpu":
		return core.BackendWebGPU, nil
	case "native_cpu", "nativecpu", "cpp":
		return core.BackendNativeCPU, nil
	case "cuda":
		return core.BackendCuda, nil
	case "metal":
		return core.BackendMetal, nil
	case "rocm", "hip":
		return core.BackendRocm, nil
	case "vulkan", "vk":
		return core.BackendVulkan, nil
	case "qnn", "hexagon", "htp":
		return core.BackendQNN, nil
	case "mlx":
		return core.BackendMLX, nil
	default:
		return 0, fmt.Errorf("accel: unknown backend %q", name)
	}
}

// Ensure installs routing (cuda/metal/rocm/vulkan) or checks availability
// (native_cpu / qnn / mlx). CPU tiled / SIMD / WebGPU are always OK.
func Ensure(b core.Backend) error {
	switch b {
	case core.BackendCPUTiled, core.BackendSIMD, core.BackendWebGPU:
		return nil
	case core.BackendCuda:
		return ensureRouter(b, func() (closer, error) { return nativecuda.EnableRouting() })
	case core.BackendMetal:
		return ensureRouter(b, func() (closer, error) { return nativemetal.EnableRouting() })
	case core.BackendRocm:
		return ensureRouter(b, func() (closer, error) { return nativerocm.EnableRouting() })
	case core.BackendVulkan:
		return ensureRouter(b, func() (closer, error) { return nativevulkan.EnableRouting() })
	case core.BackendNativeCPU:
		if core.PreferNativeCPU() != core.BackendNativeCPU {
			return fmt.Errorf("accel: native_cpu unavailable (libwelvet not loaded)")
		}
		return nil
	case core.BackendQNN:
		if !nativeqnn.Available() {
			return fmt.Errorf("accel: qnn unavailable on this host")
		}
		return nil
	case core.BackendMLX:
		if !nativemlx.Available() {
			return fmt.Errorf("accel: mlx unavailable on this host")
		}
		return nil
	default:
		return fmt.Errorf("accel: unsupported backend %s", b)
	}
}

func ensureRouter(b core.Backend, open func() (closer, error)) error {
	mu.Lock()
	defer mu.Unlock()
	if _, ok := routers[b]; ok {
		return nil
	}
	r, err := open()
	if err != nil {
		lastNote.Store(b, err.Error())
		return err
	}
	routers[b] = r
	lastNote.Store(b, "routing enabled")
	return nil
}

// Close shuts down all accel routers. Safe to call multiple times.
func Close() {
	mu.Lock()
	defer mu.Unlock()
	for b, r := range routers {
		_ = r.Close()
		delete(routers, b)
	}
}

// Probe lists every known backend and whether this process can attempt it.
func Probe() []Info {
	out := []Info{
		{ID: "cpu_tiled", Backend: int(core.BackendCPUTiled), Available: true},
		{ID: "simd", Backend: int(core.BackendSIMD), Available: true},
		{ID: "webgpu", Backend: int(core.BackendWebGPU), Available: true, Note: "requires WebGPU device at run time"},
		probeNativeCPU(),
		probeRouted("cuda", core.BackendCuda, nativecuda.LibraryPresent, "linux/windows amd64 + libwelvet_cuda"),
		probeRouted("metal", core.BackendMetal, nativemetal.LibraryPresent, "darwin arm64 + libwelvet_metal"),
		probeRouted("rocm", core.BackendRocm, nativerocm.LibraryPresent, "linux amd64 + libwelvet_rocm + ROCm"),
		probeRouted("vulkan", core.BackendVulkan, nativevulkan.LibraryPresent, "linux amd64 + libwelvet_vulkan"),
		{
			ID: "qnn", Backend: int(core.BackendQNN), Available: nativeqnn.Available(),
			Routing: "sync", Note: "needs SyncToQNN before forward (windows/android arm64)",
		},
		{
			ID: "mlx", Backend: int(core.BackendMLX), Available: nativemlx.Available(),
			Routing: "sync", Note: "needs SyncToMLX before forward (darwin arm64)",
		},
	}
	return out
}

func probeNativeCPU() Info {
	ok := core.PreferNativeCPU() == core.BackendNativeCPU
	return Info{
		ID: "native_cpu", Backend: int(core.BackendNativeCPU), Available: ok,
		Note: "welvet.cpp via nativecpp (linked libwelvet)",
	}
}

func probeRouted(id string, b core.Backend, libOK func() bool, platform string) Info {
	info := Info{ID: id, Backend: int(b), Routing: "enable", Note: platform}
	if !libOK() {
		info.Available = false
		info.Note = platform + " — library not found"
		return info
	}
	// Platform gate without permanently enabling if ensure would fail hard.
	switch b {
	case core.BackendCuda:
		info.Available = (runtime.GOOS == "linux" || runtime.GOOS == "windows") && runtime.GOARCH == "amd64"
	case core.BackendMetal:
		info.Available = runtime.GOOS == "darwin" && runtime.GOARCH == "arm64"
	case core.BackendRocm, core.BackendVulkan:
		info.Available = runtime.GOOS == "linux" && runtime.GOARCH == "amd64"
	default:
		info.Available = true
	}
	if !info.Available {
		info.Note = platform + " — wrong GOOS/GOARCH"
	}
	if v, ok := lastNote.Load(b); ok {
		info.Note = fmt.Sprintf("%s (%v)", info.Note, v)
	}
	return info
}

// NeedsSync reports whether the backend requires an explicit Sync* before forward.
func NeedsSync(b core.Backend) bool {
	return b == core.BackendQNN || b == core.BackendMLX
}
