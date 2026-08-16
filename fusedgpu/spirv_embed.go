package fusedgpu

import "embed"

// AOT SPIR-V blobs (optional). Populated by scripts/compile-hybrid-spirv.sh.
// Missing entries fall back to runtime WGSL compile.

//go:embed spirv/*.spv
var hybridSPIRVFS embed.FS

// hybridSPIRV maps pipeline name → SPIR-V bytes (4-byte words as little-endian bytes).
var hybridSPIRV = loadHybridSPIRV()

func loadHybridSPIRV() map[string][]byte {
	out := map[string][]byte{}
	ents, err := hybridSPIRVFS.ReadDir("spirv")
	if err != nil {
		return out
	}
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if len(name) < 5 || name[len(name)-4:] != ".spv" {
			continue
		}
		data, err := hybridSPIRVFS.ReadFile("spirv/" + name)
		if err != nil || len(data) < 4 {
			continue
		}
		out[name[:len(name)-4]] = data
	}
	return out
}
