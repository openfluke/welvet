package nativeqnn

import (
	"fmt"
	"sync"

	"github.com/openfluke/welvet/weights"
)

// WeightMirror holds a user-pushed snapshot of Go weights for QNN forward.
// Forward-only HTP does not mutate W; SyncFromQNN writes the snapshot back
// for API symmetry / after the user edits Go stores independently.
//
// Flow (user-owned):
//
//	SyncToQNN()   // copy Go → mirror; required before BackendQNN forward
//	Forward…     // uses mirror buffers
//	SyncFromQNN() // optional: mirror → Go Stores
type WeightMirror struct {
	mu     sync.Mutex
	stores []*weights.Store
	bufs   [][]float32 // parallel to stores
	bias   []float32   // optional Dense bias (float32)
	eps    float32     // norms
	synced bool
	layer  string // caps name: dense, rmsnorm, …
}

// Synced reports whether SyncToQNN has captured buffers.
func (m *WeightMirror) Synced() bool {
	if m == nil {
		return false
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.synced
}

// Clear drops the snapshot (next ForwardQNN will fail until SyncToQNN).
func (m *WeightMirror) Clear() {
	if m == nil {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.bufs = nil
	m.bias = nil
	m.stores = nil
	m.synced = false
}

// CaptureStores copies FlattenF32 from each store into the mirror.
func (m *WeightMirror) CaptureStores(layer string, stores ...*weights.Store) error {
	if m == nil {
		return fmt.Errorf("nativeqnn: nil mirror")
	}
	if err := RequireLayer(layer, "float32"); err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.layer = layer
	m.stores = append([]*weights.Store(nil), stores...)
	m.bufs = make([][]float32, len(stores))
	for i, s := range stores {
		if s == nil {
			return fmt.Errorf("nativeqnn: CaptureStores nil store %d", i)
		}
		flat, err := s.FlattenF32()
		if err != nil {
			return err
		}
		m.bufs[i] = append([]float32(nil), flat...)
	}
	m.synced = true
	return nil
}

// CaptureDense copies W + bias (bias may be empty).
func (m *WeightMirror) CaptureDense(W *weights.Store, biasF64 []float64) error {
	if err := m.CaptureStores("dense", W); err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(biasF64) == 0 {
		m.bias = nil
		return nil
	}
	m.bias = make([]float32, len(biasF64))
	for i, v := range biasF64 {
		m.bias[i] = float32(v)
	}
	return nil
}

// CaptureNorm copies gamma (and optional beta) plus eps.
func (m *WeightMirror) CaptureNorm(layer string, eps float32, gamma *weights.Store, beta *weights.Store) error {
	var stores []*weights.Store
	stores = append(stores, gamma)
	if beta != nil {
		stores = append(stores, beta)
	}
	if err := m.CaptureStores(layer, stores...); err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.eps = eps
	return nil
}

// MarkSynced marks a weightless layer (softmax) as ready after caps check.
func (m *WeightMirror) MarkSynced(layer string) error {
	if m == nil {
		return fmt.Errorf("nativeqnn: nil mirror")
	}
	if err := RequireLayer(layer, "float32"); err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.layer = layer
	m.synced = true
	return nil
}

// Buf returns a copy of mirrored buffer i (0=W/gamma, 1=beta…).
func (m *WeightMirror) Buf(i int) ([]float32, error) {
	if m == nil {
		return nil, fmt.Errorf("nativeqnn: nil mirror")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.synced {
		return nil, fmt.Errorf("nativeqnn: SyncToQNN required before forward")
	}
	if i < 0 || i >= len(m.bufs) {
		return nil, fmt.Errorf("nativeqnn: buf index")
	}
	return m.bufs[i], nil
}

// Bias returns mirrored Dense bias (may be nil).
func (m *WeightMirror) Bias() ([]float32, error) {
	if m == nil {
		return nil, fmt.Errorf("nativeqnn: nil mirror")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.synced {
		return nil, fmt.Errorf("nativeqnn: SyncToQNN required before forward")
	}
	return m.bias, nil
}

// Eps returns mirrored norm epsilon.
func (m *WeightMirror) Eps() (float32, error) {
	if m == nil {
		return 0, fmt.Errorf("nativeqnn: nil mirror")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.synced {
		return 0, fmt.Errorf("nativeqnn: SyncToQNN required")
	}
	return m.eps, nil
}

// SyncToGo writes mirrored buffers back into the captured Go stores.
func (m *WeightMirror) SyncToGo() error {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.synced {
		return nil
	}
	for i, s := range m.stores {
		if s == nil || i >= len(m.bufs) {
			continue
		}
		if err := s.SetFromF32(m.bufs[i]); err != nil {
			return fmt.Errorf("nativeqnn: SyncFromQNN store %d: %w", i, err)
		}
	}
	if len(m.bias) > 0 && len(m.stores) > 0 && m.stores[0] != nil {
		b := make([]float64, len(m.bias))
		for i, v := range m.bias {
			b[i] = float64(v)
		}
		m.stores[0].Bias = b
	}
	return nil
}
