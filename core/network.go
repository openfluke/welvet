package core

// Layer is the slim volumetric cell. Weight payloads live in weights.Store
// (injected by pointer from the weights package to avoid import cycles later).
// For v0, Dense owns its own weight slices; Network holds topology only.
type Layer struct {
	Type       LayerType
	DType      DType
	Activation ActivationType

	// Geometry (dense / linear)
	InputHeight  int
	OutputHeight int

	// Volumetric coordinates
	Z, Y, X, L int

	// Exec
	TileSize    int
	MultiCore   bool
	IsDisabled  bool
}

// Network is the volumetric container (Depth × Rows × Cols × LayersPerCell later).
type Network struct {
	Layers []Layer
	Exec   ExecConfig

	// NativeExact forces native dtype / QuantFormat kernels (Welvet default: true).
	// Unlike loom/poly QAT morph, there is no "fake quant" training mode.
	NativeExact bool
}

// NewNetwork returns an empty network with sane defaults.
func NewNetwork() *Network {
	return &Network{
		NativeExact: true,
		Exec: ExecConfig{
			Backend:   BackendCPUTiled,
			MultiCore: true,
			TileSize:  32,
		},
	}
}
