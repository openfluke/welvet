package entity

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/weights"
)

// RepackOpts controls .entity → .entity morphing of numerical type and/or quant.
//
// Conversion hubs through f32 (lossy). JSON sidecars are always copied as-is.
// Every other blob is re-encoded — any layer path (transformer dense/GDN,
// wav2vec2 convs, embeds, norms) as long as shape can be resolved.
type RepackOpts struct {
	Format   quant.Format // packed quant, or FormatNone for native dtype storage
	DType    *core.DType  // FormatNone element type (nil = Float32); ignored when Format != none
	Progress ProgressFunc
}

// Repack re-encodes srcPath into dstPath with the requested format/dtype.
func Repack(srcPath, dstPath string, opts RepackOpts) error {
	if opts.Format != quant.FormatNone && !quant.Supported(opts.Format) {
		return fmt.Errorf("entity.Repack: unsupported format %s", opts.Format.String())
	}
	dt := core.DTypeFloat32
	if opts.DType != nil {
		dt = *opts.DType
	}
	if opts.Format != quant.FormatNone {
		dt = core.DTypeFloat32
	}

	ef, err := Open(srcPath)
	if err != nil {
		return err
	}
	defer ef.Close()
	hdr := ef.Header()
	if hdr == nil {
		return fmt.Errorf("entity.Repack: empty header")
	}
	if hdr.Transformer == nil && hdr.Wav2Vec2 == nil {
		return fmt.Errorf("entity.Repack: unsupported entity (need transformer or wav2vec2)")
	}

	if err := os.MkdirAll(filepath.Dir(dstPath), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(dstPath), ".entity-repack-*")
	if err != nil {
		return err
	}
	payloadPath := tmp.Name()
	defer os.Remove(payloadPath)
	acc := &payloadAcc{w: tmp}
	var outBlobs []WeightBlob
	nBlobs := len(hdr.Blobs)

	for i, blob := range hdr.Blobs {
		if opts.Progress != nil {
			opts.Progress(i+1, nBlobs, blob.Path)
		}
		if strings.EqualFold(blob.DType, "JSON") {
			if err := copyBlobRaw(ef, acc, &outBlobs, blob); err != nil {
				_ = tmp.Close()
				return err
			}
			continue
		}
		f32, rows, cols, shape, err := loadAnyNumeric(ef, blob, hdr.Transformer)
		if err != nil {
			_ = tmp.Close()
			return fmt.Errorf("%s: %w", blob.Path, err)
		}
		path := normalizeWeightPath(blob.Path, opts.Format)
		if err := appendEncoded(acc, &outBlobs, path, f32, rows, cols, shape, opts.Format, dt); err != nil {
			_ = tmp.Close()
			return fmt.Errorf("%s: %w", path, err)
		}
	}

	if err := tmp.Close(); err != nil {
		return err
	}
	payload, err := os.ReadFile(payloadPath)
	if err != nil {
		return err
	}

	dtypeName := strings.ToUpper(dt.String())
	if opts.Format != quant.FormatNone {
		dtypeName = "PACKED"
	} else if dt == core.DTypeFloat32 {
		dtypeName = "FLOAT32"
	} else if dt == core.DTypeFloat16 {
		dtypeName = "FLOAT16"
	} else if dt == core.DTypeBFloat16 {
		dtypeName = "BFLOAT16"
	} else if dt == core.DTypeFloat64 {
		dtypeName = "FLOAT64"
	}

	if hdr.Wav2Vec2 != nil && hdr.Transformer == nil {
		spec := cloneWav2Vec2Spec(hdr.Wav2Vec2)
		spec.WeightDType = dtypeName
		if opts.Format == quant.FormatNone {
			spec.PackFormat = "none"
		} else {
			spec.PackFormat = opts.Format.String()
		}
		return WriteWav2Vec2File(dstPath, spec, outBlobs, payload)
	}

	spec := cloneTransformerSpec(hdr.Transformer)
	spec.WeightDType = dtypeName
	if opts.Format == quant.FormatNone {
		spec.PackFormat = ""
		spec.LMHeadPacked = false
	} else {
		spec.PackFormat = opts.Format.String()
		spec.LMHeadPacked = true
	}
	return WriteTransformerFile(dstPath, spec, outBlobs, payload)
}

func loadAnyNumeric(ef *File, blob WeightBlob, spec *TransformerSpec) (f32 []float32, rows, cols int, shape []int, err error) {
	format := quant.ParseFormatName(blob.Format)
	if format != quant.FormatNone || strings.EqualFold(blob.DType, "PACKED") {
		qb, e := ef.LoadQuantBlob(blob.Path)
		if e != nil {
			return nil, 0, 0, nil, e
		}
		f32, e = quant.Unpack(qb)
		if e != nil {
			return nil, 0, 0, nil, e
		}
		shape = append([]int(nil), blob.Shape...)
		return f32, qb.Rows, qb.Cols, shape, nil
	}

	raw, e := ef.LoadBlobBytes(blob.Path)
	if e != nil {
		return nil, 0, 0, nil, e
	}
	dt := parseBlobDType(blob.DType)
	nGuess := guessElemCount(dt, len(raw))
	f32, e = weights.DecodeNative(dt, raw, blob.Scale, nGuess)
	if e != nil {
		// Fall back to LoadBlob for legacy FLOAT32/F16/BF16/F64 aliases.
		f32, e = ef.LoadBlob(blob.Path)
		if e != nil {
			return nil, 0, 0, nil, e
		}
	}
	rows, cols, shape, e = resolveShape(blob, spec, len(f32))
	if e != nil {
		return nil, 0, 0, nil, e
	}
	return f32, rows, cols, shape, nil
}

func parseBlobDType(s string) core.DType {
	u := strings.ToUpper(strings.TrimSpace(s))
	switch u {
	case "", "FLOAT32", "F32":
		return core.DTypeFloat32
	case "FLOAT64", "F64", "DOUBLE":
		return core.DTypeFloat64
	case "FLOAT16", "F16", "FP16":
		return core.DTypeFloat16
	case "BFLOAT16", "BF16":
		return core.DTypeBFloat16
	case "PACKED", "JSON":
		return core.DTypeFloat32
	default:
		return core.ParseDType(s)
	}
}

func guessElemCount(dt core.DType, nbytes int) int {
	switch dt {
	case core.DTypeFloat64:
		return nbytes / 8
	case core.DTypeFloat16, core.DTypeBFloat16:
		return nbytes / 2
	case core.DTypeFloat32:
		return nbytes / 4
	default:
		// Low-bit packs: element count comes from rows*cols in the header.
		// When unknown, DecodeNative will fail and LoadBlob fallback runs.
		if nbytes%4 == 0 {
			return nbytes / 4
		}
		return nbytes
	}
}

func resolveShape(blob WeightBlob, spec *TransformerSpec, n int) (rows, cols int, shape []int, err error) {
	if blob.Rows > 0 && blob.Cols > 0 {
		if blob.Rows*blob.Cols != n {
			return 0, 0, nil, fmt.Errorf("rows/cols %dx%d != len %d", blob.Rows, blob.Cols, n)
		}
		return blob.Rows, blob.Cols, append([]int(nil), blob.Shape...), nil
	}
	if len(blob.Shape) > 0 {
		prod := 1
		for _, d := range blob.Shape {
			if d <= 0 {
				return 0, 0, nil, fmt.Errorf("bad shape %v", blob.Shape)
			}
			prod *= d
		}
		if prod != n {
			return 0, 0, nil, fmt.Errorf("shape %v product %d != len %d", blob.Shape, prod, n)
		}
		shape = append([]int(nil), blob.Shape...)
		if len(shape) == 1 {
			return shape[0], 1, shape, nil
		}
		// Matrix view: first dim × rest flattened (works for conv 4-D too).
		cols = 1
		for _, d := range shape[1:] {
			cols *= d
		}
		return shape[0], cols, shape, nil
	}
	if r, c, ok := inferTransformerShape(blob.Path, spec, n); ok {
		return r, c, nil, nil
	}
	// Generic fallback: column vector. Pack formats accept any rows×cols with
	// rows*cols == n (Q4_0 pads the last block).
	return n, 1, nil, nil
}

func inferTransformerShape(path string, spec *TransformerSpec, n int) (rows, cols int, ok bool) {
	if spec == nil || spec.Dims == nil {
		return 0, 0, false
	}
	d := spec.Dims
	hidden := spec.HiddenSize
	qDim := d.NumHeads * d.HeadDim
	kvDim := d.NumKVHeads * d.HeadDim
	if d.QueryDim > 0 {
		qDim = d.QueryDim
	}
	if d.KVDim > 0 {
		kvDim = d.KVDim
	}
	inter := d.IntermediateSize
	base := path
	if i := strings.LastIndex(base, "."); i >= 0 {
		base = base[i+1:]
	}
	base = strings.TrimSuffix(base, ".packed")

	try := func(r, c int) bool {
		return r > 0 && c > 0 && r*c == n
	}
	switch {
	case strings.HasSuffix(path, "lm_head") || strings.HasSuffix(path, "lm_head.packed"):
		rows, cols = spec.VocabSize, hidden
	case strings.Contains(path, "embeddings"):
		rows, cols = spec.VocabSize, hidden
	case base == "q", base == "q_proj", strings.HasSuffix(path, ".q"):
		rows, cols = qDim, hidden
	case base == "k", base == "k_proj", strings.HasSuffix(path, ".k"):
		rows, cols = kvDim, hidden
	case base == "v", base == "v_proj", strings.HasSuffix(path, ".v"):
		rows, cols = kvDim, hidden
	case base == "o", base == "o_proj", base == "out_proj", strings.HasSuffix(path, ".o"):
		rows, cols = hidden, qDim
	case base == "gate", base == "gate_proj", base == "up", base == "up_proj":
		rows, cols = inter, hidden
	case base == "down", base == "down_proj":
		rows, cols = hidden, inter
	case base == "in_proj":
		// Hybrid GDN fused in-proj — treat as (n/hidden)×hidden when divisible.
		if hidden > 0 && n%hidden == 0 {
			rows, cols = n/hidden, hidden
		}
	case strings.HasSuffix(base, "_norm"), base == "norm", strings.Contains(path, "final_norm"),
		base == "q_norm", base == "k_norm", base == "gdn_norm", base == "gdn_A_log",
		base == "gdn_dt_bias", base == "gdn_conv":
		rows, cols = n, 1
	default:
		return 0, 0, false
	}
	if !try(rows, cols) {
		return 0, 0, false
	}
	return rows, cols, true
}

func normalizeWeightPath(path string, format quant.Format) string {
	if format != quant.FormatNone {
		if path == "transformer.lm_head" {
			return "transformer.lm_head.packed"
		}
		return path
	}
	if path == "transformer.lm_head.packed" {
		return "transformer.lm_head"
	}
	return path
}

func appendEncoded(acc *payloadAcc, blobs *[]WeightBlob, path string, data []float32, rows, cols int, shape []int, format quant.Format, dt core.DType) error {
	if rows <= 0 || cols <= 0 || rows*cols != len(data) {
		return fmt.Errorf("bad encode shape %dx%d len=%d", rows, cols, len(data))
	}
	if format != quant.FormatNone {
		s, err := weights.New(rows, cols, data, core.DTypeFloat32, quant.FormatNone)
		if err != nil {
			return err
		}
		if err := weights.Convert(s, weights.ConvertOpts{Format: format}); err != nil {
			return err
		}
		if s.Packed == nil {
			return fmt.Errorf("pack produced empty blob")
		}
		wire := EncodePackedBlob(s.Packed)
		off := acc.offset
		if _, err := acc.w.Write(wire); err != nil {
			return err
		}
		acc.offset += uint64(len(wire))
		*blobs = append(*blobs, WeightBlob{
			Path: path, Offset: off, Length: uint64(len(wire)),
			DType: "PACKED", Format: format.String(),
			Rows: rows, Cols: cols, Shape: append([]int(nil), shape...), Native: true,
		})
		return nil
	}

	raw, scale, err := weights.EncodeNative(dt, data)
	if err != nil {
		return err
	}
	off := acc.offset
	if _, err := acc.w.Write(raw); err != nil {
		return err
	}
	acc.offset += uint64(len(raw))
	dtype := dtypeHeaderName(dt)
	*blobs = append(*blobs, WeightBlob{
		Path: path, Offset: off, Length: uint64(len(raw)),
		DType: dtype, Format: "none",
		Rows: rows, Cols: cols, Shape: append([]int(nil), shape...),
		Scale: scale, Native: true,
	})
	return nil
}

func dtypeHeaderName(dt core.DType) string {
	switch dt {
	case core.DTypeFloat32:
		return "FLOAT32"
	case core.DTypeFloat64:
		return "FLOAT64"
	case core.DTypeFloat16:
		return "FLOAT16"
	case core.DTypeBFloat16:
		return "BFLOAT16"
	default:
		return strings.ToUpper(dt.String())
	}
}

func copyBlobRaw(ef *File, acc *payloadAcc, blobs *[]WeightBlob, blob WeightBlob) error {
	raw, err := ef.LoadBlobBytes(blob.Path)
	if err != nil {
		return err
	}
	off := acc.offset
	if _, err := acc.w.Write(raw); err != nil {
		return err
	}
	acc.offset += uint64(len(raw))
	nb := blob
	nb.Offset = off
	nb.Length = uint64(len(raw))
	*blobs = append(*blobs, nb)
	return nil
}

func cloneTransformerSpec(s *TransformerSpec) *TransformerSpec {
	if s == nil {
		return nil
	}
	out := *s
	if s.Dims != nil {
		d := *s.Dims
		if len(s.Dims.LayerTypes) > 0 {
			d.LayerTypes = append([]string(nil), s.Dims.LayerTypes...)
		}
		out.Dims = &d
	}
	return &out
}

func cloneWav2Vec2Spec(s *Wav2Vec2Spec) *Wav2Vec2Spec {
	if s == nil {
		return nil
	}
	out := *s
	return &out
}
