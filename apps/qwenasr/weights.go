package qwenasr

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/openfluke/welvet/model/hf"
)

// tensorStore routes tensors to the correct safetensors shard. Tensors are
// decoded on demand, keeping snapshot loading practical for large checkpoints.
type tensorStore struct {
	dir   string
	shard map[string]string
}

func openTensorStore(dir string) (*tensorStore, error) {
	s := &tensorStore{dir: dir, shard: map[string]string{}}
	single := filepath.Join(dir, "model.safetensors")
	if _, err := os.Stat(single); err == nil {
		return s, nil
	}
	b, err := os.ReadFile(filepath.Join(dir, "model.safetensors.index.json"))
	if err != nil {
		return nil, fmt.Errorf("qwenasr weights: neither model.safetensors nor index: %w", err)
	}
	var index struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(b, &index); err != nil || len(index.WeightMap) == 0 {
		return nil, fmt.Errorf("qwenasr weights index: %v", err)
	}
	s.shard = index.WeightMap
	return s, nil
}
func (s *tensorStore) path(name string) (string, error) {
	if len(s.shard) == 0 {
		return filepath.Join(s.dir, "model.safetensors"), nil
	}
	p, ok := s.shard[name]
	if !ok {
		return "", fmt.Errorf("missing tensor %s", name)
	}
	return filepath.Join(s.dir, p), nil
}
func (s *tensorStore) tensor(name string) ([]float32, []int, error) {
	p, err := s.path(name)
	if err != nil {
		return nil, nil, err
	}
	m, err := hf.LoadSafetensorsWithMeta(p, func(k string) bool { return k == name })
	if err != nil {
		return nil, nil, err
	}
	t, ok := m[name]
	if !ok {
		return nil, nil, fmt.Errorf("missing tensor %s in %s", name, p)
	}
	return t.Data, t.Shape, nil
}
func (s *tensorStore) loadVec(name string) ([]float32, error) { v, _, e := s.tensor(name); return v, e }
func (s *tensorStore) loadLinear(name string, bias bool) (*Linear, error) {
	w, shape, err := s.tensor(name + ".weight")
	if err != nil {
		return nil, err
	}
	if len(shape) != 2 {
		return nil, fmt.Errorf("%s.weight: expected 2-D, got %v", name, shape)
	}
	l := &Linear{Out: shape[0], In: shape[1], W: w}
	if bias {
		if l.B, err = s.loadVec(name + ".bias"); err != nil {
			return nil, err
		}
	}
	return l, nil
}

type Conv2d struct {
	Out, In, KH, KW int
	W, B            []float32
}

func (s *tensorStore) loadConv2d(name string) (*Conv2d, error) {
	w, sh, err := s.tensor(name + ".weight")
	if err != nil {
		return nil, err
	}
	if len(sh) != 4 {
		return nil, fmt.Errorf("%s.weight: expected 4-D, got %v", name, sh)
	}
	b, err := s.loadVec(name + ".bias")
	if err != nil {
		return nil, err
	}
	return &Conv2d{sh[0], sh[1], sh[2], sh[3], w, b}, nil
}
