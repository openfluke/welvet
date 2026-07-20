package serialization

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"sort"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/model/entity"
)

// SerializeEntity writes a Grid to the native .entity wire (topology + raw blobs).
func SerializeEntity(g *architecture.Grid) ([]byte, error) {
	spec, err := BuildSpec(g)
	if err != nil {
		return nil, err
	}
	// Deep-copy topology before stripping — Layers slice must not share Stores with spec.
	topo := *spec
	topo.Layers = append([]LayerSpec(nil), spec.Layers...)
	StripWeights(&topo)

	var payload bytes.Buffer
	blobs := make([]entity.WeightBlob, 0)
	for i, ls := range spec.Layers {
		base := fmt.Sprintf("layers.%d", i)
		for _, st := range ls.Stores {
			raw, err := base64.StdEncoding.DecodeString(st.Data)
			if err != nil {
				return nil, err
			}
			off := payload.Len()
			payload.Write(raw)
			blobs = append(blobs, entity.WeightBlob{
				Path:   base + "." + st.Name,
				Offset: uint64(off),
				Length: uint64(len(raw)),
				DType:  st.DType,
				Format: st.Format,
				Rows:   st.Rows,
				Cols:   st.Cols,
				Scale:  st.Scale,
				Native: true,
			})
			if st.Bias != "" {
				br, err := base64.StdEncoding.DecodeString(st.Bias)
				if err != nil {
					return nil, err
				}
				boff := payload.Len()
				payload.Write(br)
				blobs = append(blobs, entity.WeightBlob{
					Path:   base + "." + st.Name + ".bias",
					Offset: uint64(boff),
					Length: uint64(len(br)),
					DType:  "float64",
					Format: "none",
					Native: true,
				})
			}
		}
		if len(ls.Extras) > 0 {
			keys := make([]string, 0, len(ls.Extras))
			for name := range ls.Extras {
				keys = append(keys, name)
			}
			sort.Strings(keys)
			for _, name := range keys {
				raw, err := base64.StdEncoding.DecodeString(ls.Extras[name])
				if err != nil {
					return nil, err
				}
				off := payload.Len()
				payload.Write(raw)
				blobs = append(blobs, entity.WeightBlob{
					Path:   base + ".extra." + name,
					Offset: uint64(off),
					Length: uint64(len(raw)),
					DType:  "float32",
					Format: "none",
					Native: true,
				})
			}
		}
	}
	return entity.SerializeNetwork(topo, blobs, payload.Bytes())
}

// DeserializeEntity reconstructs a Grid from .entity bytes.
func DeserializeEntity(data []byte) (*architecture.Grid, error) {
	doc, payload, err := entity.ParseNetwork(data)
	if err != nil {
		return nil, err
	}
	var spec NetworkSpec
	if err := json.Unmarshal(doc.Network, &spec); err != nil {
		return nil, fmt.Errorf("serialization: entity network: %w", err)
	}
	byPath := map[string][]byte{}
	meta := map[string]entity.WeightBlob{}
	for _, b := range doc.Blobs {
		start := int(b.Offset)
		end := start + int(b.Length)
		if end > len(payload) {
			return nil, fmt.Errorf("serialization: entity blob %q out of range", b.Path)
		}
		byPath[b.Path] = append([]byte(nil), payload[start:end]...)
		meta[b.Path] = b
	}
	for i := range spec.Layers {
		base := fmt.Sprintf("layers.%d", i)
		prefix := base + "."
		extras := map[string]string{}
		var stores []StoreBlob
		for path, raw := range byPath {
			if len(path) < len(prefix) || path[:len(prefix)] != prefix {
				continue
			}
			rest := path[len(prefix):]
			if len(rest) > 6 && rest[:6] == "extra." {
				extras[rest[6:]] = base64.StdEncoding.EncodeToString(raw)
				continue
			}
			if len(rest) > 5 && rest[len(rest)-5:] == ".bias" {
				continue
			}
			mb := meta[path]
			sb := StoreBlob{
				Name:   rest,
				DType:  mb.DType,
				Format: mb.Format,
				Rows:   mb.Rows,
				Cols:   mb.Cols,
				Data:   base64.StdEncoding.EncodeToString(raw),
				Scale:  mb.Scale,
				Native: true,
			}
			if br, ok := byPath[path+".bias"]; ok {
				sb.Bias = base64.StdEncoding.EncodeToString(br)
			}
			stores = append(stores, sb)
		}
		sort.Slice(stores, func(a, b int) bool { return stores[a].Name < stores[b].Name })
		spec.Layers[i].Stores = stores
		if len(extras) > 0 {
			spec.Layers[i].Extras = extras
		}
	}
	return GridFromSpec(&spec)
}

// SaveEntity writes path as .entity.
func SaveEntity(path string, g *architecture.Grid) error {
	data, err := SerializeEntity(g)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// LoadEntity reads a .entity checkpoint.
func LoadEntity(path string) (*architecture.Grid, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return DeserializeEntity(data)
}

// SaveGridJSON writes SerializeGrid JSON to path.
func SaveGridJSON(path string, g *architecture.Grid) error {
	data, err := SerializeGrid(g)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// LoadGridJSON reads DeserializeGrid JSON from path.
func LoadGridJSON(path string) (*architecture.Grid, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return DeserializeGrid(data)
}
