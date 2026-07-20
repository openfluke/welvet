package grouping

import (
	"strings"

	"github.com/openfluke/welvet/core"
)

// DetectedTensor represents a tensor found in a model file.
type DetectedTensor struct {
	Name    string
	Shape   []int
	DType   string
	InSize  int
	OutSize int
	CanLoad bool
}

// ArchetypeHint is a lightweight layer guess from grouped tensors.
type ArchetypeHint struct {
	Type     core.LayerType
	DModel   int
	NumHeads int
}

// GroupRelatedTensors identifies groups of tensors that belong to the same complex layer.
func GroupRelatedTensors(detected []DetectedTensor) map[string][]DetectedTensor {
	groups := make(map[string][]DetectedTensor)
	for _, d := range detected {
		if d.CanLoad {
			continue
		}
		parts := strings.Split(d.Name, ".")
		if len(parts) < 3 {
			continue
		}
		prefix := ""
		for i, p := range parts {
			if strings.HasPrefix(p, "layer") || strings.EqualFold(p, "layers") {
				if i+2 < len(parts) {
					prefix = strings.Join(parts[:i+3], ".")
					break
				}
			}
		}
		if prefix != "" {
			groups[prefix] = append(groups[prefix], d)
		}
	}
	return groups
}

func tensorHas(tensors []DetectedTensor, keys ...string) bool {
	found := make(map[string]bool)
	for _, t := range tensors {
		n := strings.ToLower(t.Name)
		for _, k := range keys {
			if strings.Contains(n, k) {
				found[k] = true
			}
		}
	}
	for _, k := range keys {
		if !found[k] {
			return false
		}
	}
	return true
}

// DetectSwiGLU returns true when gate/up/down tensors are present.
func DetectSwiGLU(_ string, tensors []DetectedTensor, dModel int) (bool, ArchetypeHint) {
	ok := tensorHas(tensors, "gate", "up", "down") || tensorHas(tensors, "w1", "w2", "w3")
	if !ok {
		return false, ArchetypeHint{}
	}
	return true, ArchetypeHint{Type: core.LayerSwiGLU, DModel: dModel}
}

// DetectMHA returns true when Q/K/V/O projections are present.
func DetectMHA(_ string, tensors []DetectedTensor, dModel, numHeads int) (bool, ArchetypeHint) {
	ok := tensorHas(tensors, "q_proj", "k_proj", "v_proj", "o_proj") ||
		tensorHas(tensors, "query", "key", "value", "out_proj")
	if !ok {
		return false, ArchetypeHint{}
	}
	return true, ArchetypeHint{Type: core.LayerMultiHeadAttention, DModel: dModel, NumHeads: numHeads}
}

// DetectRMSNorm returns true when a norm weight tensor is present.
func DetectRMSNorm(_ string, tensors []DetectedTensor, dModel int) (bool, ArchetypeHint) {
	for _, t := range tensors {
		n := strings.ToLower(t.Name)
		if strings.Contains(n, "norm") || strings.Contains(n, "weight") {
			return true, ArchetypeHint{Type: core.LayerRMSNorm, DModel: dModel}
		}
	}
	return false, ArchetypeHint{}
}

// DetectLayerNorm returns true when gamma/beta style weights are present.
func DetectLayerNorm(_ string, tensors []DetectedTensor, dModel int) (bool, ArchetypeHint) {
	for _, t := range tensors {
		n := strings.ToLower(t.Name)
		if strings.Contains(n, "weight") || strings.Contains(n, "gamma") {
			return true, ArchetypeHint{Type: core.LayerLayerNorm, DModel: dModel}
		}
	}
	return false, ArchetypeHint{}
}

// DetectCNN returns true when a conv weight tensor is present.
func DetectCNN(_ string, tensors []DetectedTensor, ltype core.LayerType) (bool, ArchetypeHint) {
	for _, t := range tensors {
		if strings.Contains(strings.ToLower(t.Name), "weight") {
			return true, ArchetypeHint{Type: ltype}
		}
	}
	return false, ArchetypeHint{}
}
