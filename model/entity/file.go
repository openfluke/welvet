package entity

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
)

// WeightBlob indexes one payload in the blob section (F32 or packed quant).
type WeightBlob struct {
	Path   string  `json:"path"`
	Offset uint64  `json:"offset"`
	Length uint64  `json:"length"`
	DType  string  `json:"dtype"`  // "FLOAT32" for norms/embed; also welvet dtype names
	Format string  `json:"format"` // "none", "Q4_0", "Q4_K", …
	Rows   int     `json:"rows,omitempty"`
	Cols   int     `json:"cols,omitempty"`
	Shape  []int   `json:"shape,omitempty"` // ND tensors (wav2vec2 / whisper)
	Scale  float32 `json:"scale,omitempty"`
	Native bool    `json:"native"`
}

// Wav2Vec2Spec is the ASR add-on in the ENTITY header (facebook/wav2vec2-* CTC).
type Wav2Vec2Spec struct {
	Architecture string `json:"architecture"`
	HiddenSize   int    `json:"hidden_size"`
	VocabSize    int    `json:"vocab_size"`
	NumLayers    int    `json:"num_hidden_layers"`
	NumHeads     int    `json:"num_attention_heads"`
	Intermediate int    `json:"intermediate_size"`
	PadTokenID   int    `json:"pad_token_id"`
	WeightDType  string `json:"weight_dtype,omitempty"`
	PackFormat   string `json:"pack_format,omitempty"`
	Snapshot     string `json:"snapshot,omitempty"`
	Repo         string `json:"repo,omitempty"`
	VocabBlob    string `json:"vocab_blob,omitempty"`
	ConfigBlob   string `json:"config_blob,omitempty"`
}

// TransformerDims records decoder hyperparameters.
type TransformerDims struct {
	NumLayers        int     `json:"num_layers"`
	NumHeads         int     `json:"num_heads"`
	NumKVHeads       int     `json:"num_kv_heads"`
	HeadDim          int     `json:"head_dim"`
	QueryDim         int     `json:"query_dim,omitempty"`
	KVDim            int     `json:"kv_dim,omitempty"`
	IntermediateSize int     `json:"intermediate_size"`
	RMSNormEps       float64 `json:"rms_norm_eps,omitempty"`
	RoPEFreqBase     float64 `json:"rope_freq_base,omitempty"`
	// Hybrid (Qwen3.5 / Bonsai)
	PartialRotaryFactor float64  `json:"partial_rotary_factor,omitempty"`
	AttnOutputGate      bool     `json:"attn_output_gate,omitempty"`
	LayerTypes          []string `json:"layer_types,omitempty"`
	LinearConvKernel    int      `json:"linear_conv_kernel,omitempty"`
	LinearNumKeyHeads   int      `json:"linear_num_key_heads,omitempty"`
	LinearNumValueHeads int      `json:"linear_num_value_heads,omitempty"`
	LinearKeyHeadDim    int      `json:"linear_key_head_dim,omitempty"`
	LinearValueHeadDim  int      `json:"linear_value_head_dim,omitempty"`
}

// TransformerSpec is the decoder add-on in the ENTITY header.
type TransformerSpec struct {
	Architecture string           `json:"architecture"`
	HiddenSize   int              `json:"hidden_size"`
	VocabSize    int              `json:"vocab_size"`
	LMHeadTied   bool             `json:"lm_head_tied,omitempty"`
	HasFinalNorm bool             `json:"has_final_norm,omitempty"`
	WeightDType  string           `json:"weight_dtype,omitempty"`
	PackFormat   string           `json:"pack_format,omitempty"` // baked decoder quant (Q4_0, Q4_K, …)
	LMHeadPacked bool             `json:"lm_head_packed,omitempty"`
	Snapshot     string           `json:"snapshot,omitempty"`
	Tokenizer    string           `json:"tokenizer,omitempty"`
	Repo         string           `json:"repo,omitempty"`
	Engine       string           `json:"engine,omitempty"`
	Dims         *TransformerDims `json:"dims,omitempty"`
}

type headerDoc struct {
	FormatVersion uint16           `json:"format_version"`
	Engine        string           `json:"engine"`
	Status        string           `json:"status"`
	Network       json.RawMessage  `json:"network,omitempty"`
	Transformer   *TransformerSpec `json:"transformer,omitempty"`
	Wav2Vec2      *Wav2Vec2Spec    `json:"wav2vec2,omitempty"`
	Blobs         []WeightBlob     `json:"blobs"`
}

// Header is parsed metadata (no weight bytes).
type Header struct {
	FormatVersion uint16
	Flags         uint16
	Transformer   *TransformerSpec
	Wav2Vec2      *Wav2Vec2Spec
	Blobs         []WeightBlob
	DataOffset    int
	Status        string
	Engine        string
}

// File is a random-access .entity reader.
type File struct {
	f   *os.File
	hdr *Header
}

// Open opens a Welvet .entity checkpoint.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	ef := &File{f: f}
	if err := ef.readHeader(); err != nil {
		_ = f.Close()
		return nil, err
	}
	return ef, nil
}

// Close releases the file handle.
func (ef *File) Close() error {
	if ef == nil || ef.f == nil {
		return nil
	}
	err := ef.f.Close()
	ef.f = nil
	return err
}

// Header returns parsed metadata.
func (ef *File) Header() *Header {
	if ef == nil {
		return nil
	}
	return ef.hdr
}

func (ef *File) readHeader() error {
	fixed := make([]byte, fixedHeaderSize())
	if _, err := ef.f.ReadAt(fixed, 0); err != nil {
		return fmt.Errorf("entity fixed header: %w", err)
	}
	if string(fixed[:8]) != Magic {
		return fmt.Errorf("invalid entity magic: %q (want ENTITY)", fixed[:8])
	}
	version := binary.LittleEndian.Uint16(fixed[8:10])
	if version != FormatVersion {
		return fmt.Errorf("unsupported entity version: %d", version)
	}
	flags := binary.LittleEndian.Uint16(fixed[10:12])
	headerLen := binary.LittleEndian.Uint64(fixed[12:20])
	if headerLen > headerMaxSize {
		return fmt.Errorf("entity header size unreasonable: %d", headerLen)
	}
	headerJSON := make([]byte, headerLen)
	if _, err := ef.f.ReadAt(headerJSON, int64(fixedHeaderSize())); err != nil {
		return fmt.Errorf("entity header JSON: %w", err)
	}
	var doc headerDoc
	if err := json.Unmarshal(headerJSON, &doc); err != nil {
		return fmt.Errorf("entity header JSON: %w", err)
	}
	ef.hdr = &Header{
		FormatVersion: version,
		Flags:         flags,
		Transformer:   doc.Transformer,
		Wav2Vec2:      doc.Wav2Vec2,
		Blobs:         doc.Blobs,
		DataOffset:    fixedHeaderSize() + int(headerLen),
		Status:        doc.Status,
		Engine:        doc.Engine,
	}
	return nil
}

// LoadBlob reads one FormatNone numeric blob and returns float32 values.
// Supports FLOAT32 / FLOAT64 / FLOAT16 / BFLOAT16 (and lowercase welvet names).
// Packed quants use LoadQuantBlob; tokenizer JSON uses LoadTokenizerJSON.
func (ef *File) LoadBlob(path string) ([]float32, error) {
	if ef == nil || ef.hdr == nil {
		return nil, fmt.Errorf("entity: not open")
	}
	blob, err := ef.findBlob(path)
	if err != nil {
		return nil, err
	}
	format := strings.ToLower(strings.TrimSpace(blob.Format))
	if format != "" && format != "none" {
		return nil, fmt.Errorf("entity blob %q: packed format %q — use LoadQuantBlob", path, blob.Format)
	}
	dtype := strings.ToUpper(strings.TrimSpace(blob.DType))
	switch dtype {
	case "PACKED":
		return nil, fmt.Errorf("entity blob %q: packed payload — use LoadQuantBlob", path)
	case "JSON":
		return nil, fmt.Errorf("entity blob %q: JSON payload — use LoadBlobBytes / LoadTokenizerJSON", path)
	}
	raw, err := ef.LoadBlobBytes(path)
	if err != nil {
		return nil, err
	}
	return decodeNumericBlob(path, dtype, raw, blob.Scale)
}

// LoadBlobBytes reads raw payload bytes for a blob path.
func (ef *File) LoadBlobBytes(path string) ([]byte, error) {
	blob, err := ef.findBlob(path)
	if err != nil {
		return nil, err
	}
	raw := make([]byte, int(blob.Length))
	off := int64(ef.hdr.DataOffset) + int64(blob.Offset)
	if _, err := ef.f.ReadAt(raw, off); err != nil {
		return nil, fmt.Errorf("entity blob %q: %w", path, err)
	}
	return raw, nil
}

// LoadQuantBlob decodes a packed quant blob (Format != none).
func (ef *File) LoadQuantBlob(path string) (*quant.Blob, error) {
	blob, err := ef.findBlob(path)
	if err != nil {
		return nil, err
	}
	format := quant.ParseFormatName(blob.Format)
	if format == quant.FormatNone {
		return nil, fmt.Errorf("entity blob %q: not a packed quant blob (format=%q)", path, blob.Format)
	}
	if blob.Rows <= 0 || blob.Cols <= 0 {
		return nil, fmt.Errorf("entity blob %q: missing rows/cols", path)
	}
	wire, err := ef.LoadBlobBytes(path)
	if err != nil {
		return nil, err
	}
	return DecodePackedBlob(format, blob.Rows, blob.Cols, wire)
}

func (ef *File) findBlob(path string) (*WeightBlob, error) {
	if ef == nil || ef.hdr == nil {
		return nil, fmt.Errorf("entity: not open")
	}
	for i := range ef.hdr.Blobs {
		if ef.hdr.Blobs[i].Path == path {
			return &ef.hdr.Blobs[i], nil
		}
	}
	return nil, fmt.Errorf("entity blob %q not found", path)
}

// LoadTokenizerJSON reads the embedded tokenizer.json blob, if present.
func (ef *File) LoadTokenizerJSON() ([]byte, error) {
	return ef.LoadBlobBytes(TokenizerBlobPath)
}

// HasTokenizerBlob reports whether the entity embeds tokenizer.json.
func (ef *File) HasTokenizerBlob() bool {
	if ef == nil || ef.hdr == nil {
		return false
	}
	for i := range ef.hdr.Blobs {
		if ef.hdr.Blobs[i].Path == TokenizerBlobPath {
			return true
		}
	}
	return false
}

// PeekMagic reads the first 8 bytes of path.
func PeekMagic(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	var buf [8]byte
	n, err := f.Read(buf[:])
	if err != nil || n < 8 {
		return "", fmt.Errorf("short read")
	}
	return string(buf[:]), nil
}

func decodeNumericBlob(path, dtype string, raw []byte, scale float32) ([]float32, error) {
	if dtype == "" {
		dtype = "FLOAT32"
	}
	dt := core.ParseDType(dtype)
	switch strings.ToUpper(dtype) {
	case "FLOAT32", "F32", "":
		dt = core.DTypeFloat32
	case "FLOAT64", "F64", "DOUBLE":
		dt = core.DTypeFloat64
	case "FLOAT16", "F16", "FP16":
		dt = core.DTypeFloat16
	case "BFLOAT16", "BF16":
		dt = core.DTypeBFloat16
	}
	switch dt {
	case core.DTypeFloat32:
		if len(raw)%4 != 0 {
			return nil, fmt.Errorf("entity blob %q: length %d not multiple of 4", path, len(raw))
		}
		n := len(raw) / 4
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4 : i*4+4]))
		}
		return out, nil
	case core.DTypeFloat64:
		if len(raw)%8 != 0 {
			return nil, fmt.Errorf("entity blob %q: length %d not multiple of 8", path, len(raw))
		}
		n := len(raw) / 8
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			bits := binary.LittleEndian.Uint64(raw[i*8 : i*8+8])
			out[i] = float32(math.Float64frombits(bits))
		}
		return out, nil
	case core.DTypeFloat16:
		if len(raw)%2 != 0 {
			return nil, fmt.Errorf("entity blob %q: length %d not multiple of 2", path, len(raw))
		}
		n := len(raw) / 2
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = core.Float16ToFloat32(binary.LittleEndian.Uint16(raw[i*2 : i*2+2]))
		}
		return out, nil
	case core.DTypeBFloat16:
		if len(raw)%2 != 0 {
			return nil, fmt.Errorf("entity blob %q: length %d not multiple of 2", path, len(raw))
		}
		n := len(raw) / 2
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = core.BFloat16ToFloat32(binary.LittleEndian.Uint16(raw[i*2 : i*2+2]))
		}
		return out, nil
	default:
		_ = scale
		return nil, fmt.Errorf("entity blob %q: unsupported FormatNone dtype %q (use LoadBlobBytes)", path, dtype)
	}
}
