package nativecpp

import (
	"fmt"
	"sync"

	"github.com/openfluke/welvet/weights"
)

// Mirror holds the latest C++ W_after (or concat of stores) until the user
// pulls it into Go weights.Store(s) for .entity save / Go-side use.
//
// Typical flow:
//
//	TrainNative…  // C++ SGD; Capture pending; Go Store unchanged
//	… more native steps (uploads use PendingOrStore) …
//	SyncToGo()    // CommitStore(s) — now safe to write .entity
type Mirror struct {
	mu     sync.Mutex
	stores []*weights.Store
	after  []float32
	dirty  bool
}

// Capture records a pending native weight buffer for later SyncToGo.
// after is copied; stores are the Go targets in concat order (Gate|Up|Down, etc.).
func (m *Mirror) Capture(after []float32, stores ...*weights.Store) {
	if m == nil {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.stores = append([]*weights.Store(nil), stores...)
	m.after = append([]float32(nil), after...)
	m.dirty = len(m.after) > 0 && len(m.stores) > 0
}

// Dirty reports whether SyncToGo would write something.
func (m *Mirror) Dirty() bool {
	if m == nil {
		return false
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.dirty
}

// SyncToGo commits pending native weights into the captured Go stores (SetFromF32).
// Safe to call when not dirty (no-op). Clears the pending buffer on success.
func (m *Mirror) SyncToGo() error {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.dirty {
		return nil
	}
	if err := CommitStores(m.after, m.stores...); err != nil {
		return fmt.Errorf("nativecpp: SyncToGo: %w", err)
	}
	m.dirty = false
	m.after = nil
	// keep store pointers so PendingOrStore can still fall through to Store
	return nil
}

// PendingOrStore returns the pending slice for store if this mirror holds it,
// otherwise FlattenF32 from the Go store (upload source for the next C ABI call).
func (m *Mirror) PendingOrStore(s *weights.Store) ([]float32, error) {
	if s == nil {
		return nil, fmt.Errorf("nativecpp: PendingOrStore nil")
	}
	if m != nil {
		m.mu.Lock()
		if m.dirty {
			off := 0
			for _, st := range m.stores {
				n := st.Rows * st.Cols
				if st == s {
					if off+n > len(m.after) {
						m.mu.Unlock()
						return nil, fmt.Errorf("nativecpp: PendingOrStore short buffer")
					}
					out := append([]float32(nil), m.after[off:off+n]...)
					m.mu.Unlock()
					return out, nil
				}
				off += n
			}
		}
		m.mu.Unlock()
	}
	return s.FlattenF32()
}

// Clear drops pending native weights without writing Go stores.
func (m *Mirror) Clear() {
	if m == nil {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.after = nil
	m.dirty = false
}
