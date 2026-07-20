package serialization

import (
	"github.com/openfluke/welvet/quant"
)

// NetworkSpec is the JSON-serializable grid state (loom PersistenceNetworkSpec).
type NetworkSpec struct {
	ID            string      `json:"id"`
	InitSeed      uint64      `json:"init_seed,omitempty"`
	Depth         int         `json:"depth"`
	Rows          int         `json:"rows"`
	Cols          int         `json:"cols"`
	LayersPerCell int         `json:"layers_per_cell"`
	Layers        []LayerSpec `json:"layers"`
}

// StoreBlob is one weight store's storage-truth snapshot (native dtype or packed quant).
// No QAT — Data is the on-disk bytes for Format×DType as held in weights.Store.
type StoreBlob struct {
	Name   string  `json:"name"`
	DType  string  `json:"dtype"`
	Format string  `json:"format"`
	Rows   int     `json:"rows"`
	Cols   int     `json:"cols"`
	Data   string  `json:"data"` // base64
	Scale  float32 `json:"scale,omitempty"`
	Bias   string  `json:"bias,omitempty"` // base64 LE float64
	Native bool    `json:"native"`
}

// LayerSpec is one cell (all Ops).
type LayerSpec struct {
	Z          int    `json:"z"`
	Y          int    `json:"y"`
	X          int    `json:"x"`
	L          int    `json:"l"`
	Type       string `json:"type"`
	Activation string `json:"activation"`
	DType      string `json:"dtype"`
	Format     string `json:"format,omitempty"` // primary store format hint

	InputHeight  int `json:"input_height,omitempty"`
	InputWidth   int `json:"input_width,omitempty"`
	InputDepth   int `json:"input_depth,omitempty"`
	OutputHeight int `json:"output_height,omitempty"`
	OutputWidth  int `json:"output_width,omitempty"`
	OutputDepth  int `json:"output_depth,omitempty"`

	InputChannels int `json:"input_channels,omitempty"`
	Filters       int `json:"filters,omitempty"`
	KernelSize    int `json:"kernel_size,omitempty"`
	Stride        int `json:"stride,omitempty"`
	Padding       int `json:"padding,omitempty"`
	OutputPadding int `json:"output_padding,omitempty"`

	NumHeads   int `json:"num_heads,omitempty"`
	NumKVHeads int `json:"num_kv_heads,omitempty"`
	DModel     int `json:"d_model,omitempty"`
	HeadDim    int `json:"head_dim,omitempty"`
	SeqLength  int `json:"seq_length,omitempty"`

	VocabSize    int `json:"vocab_size,omitempty"`
	EmbeddingDim int `json:"embedding_dim,omitempty"`

	NumClusters int     `json:"num_clusters,omitempty"`
	OutputMode  string  `json:"output_mode,omitempty"`
	CombineMode string  `json:"combine_mode,omitempty"`
	Temperature float64 `json:"temperature,omitempty"`
	SoftmaxKind     string  `json:"softmax_kind,omitempty"`
	SoftmaxRows     int     `json:"softmax_rows,omitempty"`
	SoftmaxCols     int     `json:"softmax_cols,omitempty"`
	SoftmaxMask     []bool  `json:"softmax_mask,omitempty"`
	EntmaxAlpha     float64 `json:"entmax_alpha,omitempty"`
	HierarchyLevels []int   `json:"hierarchy_levels,omitempty"`
	Eps         float64 `json:"eps,omitempty"`

	Expand          int `json:"expand,omitempty"`
	DState          int `json:"d_state,omitempty"`
	IntermediateDim int `json:"intermediate_dim,omitempty"`
	Branches        int `json:"branches,omitempty"`
	OutFeat         int `json:"out_feat,omitempty"`
	DepthN          int `json:"depth_n,omitempty"` // sequential/residual child count

	// GDN
	HiddenSize    int `json:"hidden_size,omitempty"`
	NumKeyHeads   int `json:"num_key_heads,omitempty"`
	NumValueHeads int `json:"num_value_heads,omitempty"`
	KeyHeadDim    int `json:"key_head_dim,omitempty"`
	ValueHeadDim  int `json:"value_head_dim,omitempty"`
	ConvKernel    int `json:"conv_kernel,omitempty"`

	// MHA policy (minimal)
	Mask   string `json:"mask,omitempty"`
	Causal bool   `json:"causal,omitempty"`

	IsRemoteLink bool `json:"is_remote_link,omitempty"`
	TargetZ      int  `json:"target_z,omitempty"`
	TargetY      int  `json:"target_y,omitempty"`
	TargetX      int  `json:"target_x,omitempty"`
	TargetL      int  `json:"target_l,omitempty"`
	IsDisabled   bool `json:"is_disabled,omitempty"`

	Stores []StoreBlob       `json:"stores,omitempty"`
	Extras map[string]string `json:"extras,omitempty"` // base64 LE f32 sidecars

	// Legacy Dense v0 single-blob fields (still accepted on load).
	Weights string  `json:"weights,omitempty"`
	Native  bool    `json:"native,omitempty"`
	Scale   float32 `json:"scale,omitempty"`
}

// PersistenceNetworkSpec / PersistenceLayerSpec loom aliases.
type PersistenceNetworkSpec = NetworkSpec
type PersistenceLayerSpec = LayerSpec

// PackableFormats are formats with Pack() support (AffinePacked is import-only GAP).
func PackableFormats() []quant.Format {
	out := make([]quant.Format, 0, len(quant.AllFormats)-1)
	for _, f := range quant.AllFormats {
		if f == quant.FormatAffinePacked {
			continue
		}
		out = append(out, f)
	}
	return out
}
