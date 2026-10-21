package nativecpp

import (
	"fmt"

	"github.com/openfluke/welvet/weights"
)

// CommitStore writes a C ABI W_after / gamma_after buffer back into a Go weights.Store
// (re-encodes as the store's DType/Format via SetFromF32).
func CommitStore(s *weights.Store, after []float32) error {
	if s == nil {
		return fmt.Errorf("nativecpp: CommitStore nil store")
	}
	if err := s.SetFromF32(after); err != nil {
		return fmt.Errorf("nativecpp: CommitStore: %w", err)
	}
	return nil
}

// CommitStores writes consecutive packed slices back into stores (e.g. SwiGLU Gate|Up|Down).
// Each store consumes Rows*Cols floats from after, in order.
func CommitStores(after []float32, stores ...*weights.Store) error {
	off := 0
	for i, s := range stores {
		if s == nil {
			return fmt.Errorf("nativecpp: CommitStores nil store %d", i)
		}
		n := s.Rows * s.Cols
		if off+n > len(after) {
			return fmt.Errorf("nativecpp: CommitStores short buffer need %d have %d", off+n, len(after))
		}
		if err := CommitStore(s, after[off:off+n]); err != nil {
			return err
		}
		off += n
	}
	return nil
}
