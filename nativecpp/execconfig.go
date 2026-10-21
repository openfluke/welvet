package nativecpp

import (
	"encoding/json"
	"fmt"
)

// Backend names match the execution planner (independent of weight dtype / TrainMode).
const (
	BackendInherit   = "inherit"
	BackendGoCPU     = "go_cpu"
	BackendNativeCPU = "native_cpu"
	BackendCUDA      = "cuda"
	BackendMetal     = "metal"
	BackendNPU       = "npu"
)

// ExecConfig is the typed form shared by JSON and native Go configuration.
type ExecConfig struct {
	Default string                   `json:"default"`
	Volumes map[string]VolumeBackend `json:"volumes,omitempty"`
}

type VolumeBackend struct {
	Backend string                    `json:"backend,omitempty"`
	Sectors map[string]SectorBackend  `json:"sectors,omitempty"`
}

type SectorBackend struct {
	Backend string                  `json:"backend,omitempty"`
	Device  int                     `json:"device,omitempty"`
	Layers  map[string]LayerBackend `json:"layers,omitempty"`
}

type LayerBackend struct {
	Backend string `json:"backend,omitempty"`
	Device  int    `json:"device,omitempty"`
}

// Override priority: Layer > Sector > Volume > Network(default).
func (c ExecConfig) Resolve(volume, sector, layer string) (backend string, device int) {
	backend = c.Default
	if backend == "" {
		if NativeCPUAvailable() {
			backend = BackendNativeCPU
		} else {
			backend = BackendGoCPU
		}
	}
	device = -1
	if c.Volumes == nil {
		return backend, device
	}
	vol, ok := c.Volumes[volume]
	if !ok {
		return backend, device
	}
	if vol.Backend != "" && vol.Backend != BackendInherit {
		backend = vol.Backend
	}
	if vol.Sectors == nil {
		return backend, device
	}
	sec, ok := vol.Sectors[sector]
	if !ok {
		return backend, device
	}
	if sec.Backend != "" && sec.Backend != BackendInherit {
		backend = sec.Backend
		device = sec.Device
	}
	if sec.Layers == nil {
		return backend, device
	}
	lay, ok := sec.Layers[layer]
	if !ok {
		return backend, device
	}
	if lay.Backend != "" && lay.Backend != BackendInherit {
		backend = lay.Backend
		if lay.Device != 0 || sec.Device != 0 {
			device = lay.Device
		}
	}
	return backend, device
}

func ParseExecConfigJSON(b []byte) (ExecConfig, error) {
	var c ExecConfig
	if err := json.Unmarshal(b, &c); err != nil {
		return ExecConfig{}, fmt.Errorf("nativecpp: exec config json: %w", err)
	}
	return c, nil
}

func (c ExecConfig) ToJSON() ([]byte, error) {
	return json.MarshalIndent(c, "", "  ")
}

// Available reports whether native_cpu can run on this process (lib loaded).
func NativeCPUAvailable() bool {
	return libLoaded()
}
