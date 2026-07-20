package hardware

import (
	"encoding/json"
	"fmt"
	"net"
	"os"
	"runtime"
	"runtime/debug"
)

// SystemAudit contains comprehensive hardware and OS information.
type SystemAudit struct {
	OS        OSInfo        `json:"os"`
	CPU       CPUInfo       `json:"cpu"`
	RAM       RAMInfo       `json:"ram"`
	GPU       GPUInfo       `json:"gpu"`
	Disk      []DiskInfo    `json:"disk"`
	Network   []NetworkInfo `json:"network"`
	GoRuntime GoRuntimeInfo `json:"go_runtime"`
}

type OSInfo struct {
	Hostname string `json:"hostname"`
	Platform string `json:"platform"` // GOOS
	Arch     string `json:"arch"`     // GOARCH
	Uptime   string `json:"uptime"`
}

type CPUInfo struct {
	Model      string `json:"model"`
	Logical    int    `json:"logical_cores"`
	GOMAXPROCS int    `json:"gomaxprocs"`
}

type RAMInfo struct {
	Total uint64 `json:"total_bytes"`
	Free  uint64 `json:"free_bytes"`
	Used  uint64 `json:"used_bytes"`
}

type GPUInfo struct {
	Model  string `json:"model"`
	VRAM   uint64 `json:"vram_bytes"`
	Vendor string `json:"vendor,omitempty"`
}

type DiskInfo struct {
	Path  string `json:"path"`
	Total uint64 `json:"total_bytes"`
	Free  uint64 `json:"free_bytes"`
}

type NetworkInfo struct {
	Name  string   `json:"name"`
	MAC   string   `json:"mac"`
	IPs   []string `json:"ips"`
	Flags string   `json:"flags"`
	MTU   int      `json:"mtu"`
}

type GoRuntimeInfo struct {
	Version   string           `json:"version"`
	NumGorout int              `json:"num_goroutines"`
	MemAlloc  uint64           `json:"mem_alloc_bytes"`
	BuildInfo *debug.BuildInfo `json:"build_info,omitempty"`
}

// Audit extracts as much information as possible natively without external commands.
func Audit() *SystemAudit {
	audit := &SystemAudit{}

	audit.OS.Hostname, _ = os.Hostname()
	audit.OS.Platform = runtime.GOOS
	audit.OS.Arch = runtime.GOARCH
	audit.OS.Uptime = getUptime()

	audit.CPU.Model = getCPUModel()
	audit.CPU.Logical = runtime.NumCPU()
	audit.CPU.GOMAXPROCS = runtime.GOMAXPROCS(0)

	audit.RAM.Total, audit.RAM.Free = getSystemMemory()
	audit.RAM.Used = audit.RAM.Total - audit.RAM.Free

	audit.GPU = getGPUInfo()

	audit.Disk = getDiskUsage()
	audit.Network = getNetworkInfo()

	audit.GoRuntime.Version = runtime.Version()
	audit.GoRuntime.NumGorout = runtime.NumGoroutine()
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	audit.GoRuntime.MemAlloc = ms.Alloc
	audit.GoRuntime.BuildInfo, _ = debug.ReadBuildInfo()

	return audit
}

// AuditSystem is the loom alias (Grid/WebGPU fallback omitted in v0).
func AuditSystem(_ any) *SystemAudit { return Audit() }


// ToJSON returns the audit formatted as a JSON string.
func (a *SystemAudit) ToJSON() string {
	b, err := json.MarshalIndent(a, "", "  ")
	if err != nil {
		return "{ \"error\": \"failed to marshal audit\" }"
	}
	return string(b)
}

// GetDeviceDescription returns a human-readable string summary.
func GetDeviceDescription(_ any) string {
	a := Audit()
	ramGB := float64(a.RAM.Total) / (1024 * 1024 * 1024)
	return fmt.Sprintf("OS: %s | CPU: %s (%d) | RAM: %.2f GB | GPU: %s (%.1f GB)",
		a.OS.Platform, a.CPU.Model, a.CPU.Logical, ramGB, a.GPU.Model, float64(a.GPU.VRAM)/(1024*1024*1024))
}

func getNetworkInfo() []NetworkInfo {
	var infos []NetworkInfo
	ifaces, err := net.Interfaces()
	if err == nil {
		for _, iface := range ifaces {
			ni := NetworkInfo{
				Name:  iface.Name,
				MAC:   iface.HardwareAddr.String(),
				Flags: iface.Flags.String(),
				MTU:   iface.MTU,
			}
			addrs, _ := iface.Addrs()
			for _, addr := range addrs {
				ni.IPs = append(ni.IPs, addr.String())
			}
			infos = append(infos, ni)
		}
	}
	return infos
}

// Description returns a short one-line device summary.
func Description() string {
	return GetDeviceDescription(nil)
}
