// Package nativecuda mounts Go Welvet models in the independent Linux CUDA
// backend. Mount copies weights; training never silently changes Go stores.
package nativecuda

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/gdn"
	"github.com/openfluke/welvet/layers/mamba"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/weights"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"
)

type driver interface {
	segment(int, float64, []byte) ([]byte, error)
	forward([]byte) ([]byte, error)
	train(int, float64, int, []byte) (float64, error)
	export() ([]byte, error)
	sync(int) error
	version() string
	identity() string
	close()
}

// NativeError preserves the status code returned by the native ABI.
type NativeError struct {
	Code    int
	Message string
}

func (e *NativeError) Error() string {
	return fmt.Sprintf("nativecuda: code=%d: %s", e.Code, e.Message)
}

type Options struct{ LibraryPath string }

// Session retains its model, native weights and Line/proxy state between calls.
// Calls on one session serialize. Call SyncToGo before saving the source model.
// Close discards unsynchronized updates and is idempotent.
type Session struct {
	mu      sync.Mutex
	d       driver
	binding binding
}

func defaultLibrary() string {
	if p := os.Getenv("WELVET_CUDA_LIBRARY"); p != "" {
		return p
	}
	if exe, e := os.Executable(); e == nil {
		p := filepath.Join(filepath.Dir(exe), "libwelvet_cuda.so")
		if _, e = os.Stat(p); e == nil {
			return p
		}
	}
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "..", "native", "cuda", "linux_x86_64", "libwelvet_cuda.so")
}

// Mount accepts a supported Go layer, parallel.Stack, parallel.Layer or Grid.
// Options selects the library; no CPU fallback occurs if loading/mounting fails.
func Mount(model any, options ...Options) (*Session, error) {
	if len(options) > 1 {
		return nil, fmt.Errorf("nativecuda: at most one Options value")
	}
	payload, b, e := encodeModel(model)
	if e != nil {
		return nil, e
	}
	path := defaultLibrary()
	if len(options) == 1 && options[0].LibraryPath != "" {
		path = options[0].LibraryPath
	}
	d, e := openDriver(path, payload)
	if e != nil {
		return nil, e
	}
	s := &Session{d: d, binding: b}
	runtime.SetFinalizer(s, func(s *Session) { _ = s.Close() })
	return s, nil
}
func (s *Session) ready() error {
	if s == nil || s.d == nil {
		return fmt.Errorf("nativecuda: session is closed")
	}
	return nil
}
func (s *Session) Version() string {
	if s == nil {
		return ""
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.d == nil {
		return ""
	}
	return s.d.version()
}
func (s *Session) Identity() string {
	if s == nil {
		return ""
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.d == nil {
		return ""
	}
	return s.d.identity()
}
func encodeTensor(e *encoder, t *core.Tensor[float32]) {
	if t == nil || len(t.Shape) < 1 || len(t.Shape) > 5 {
		e.fail("tensor rank must be 1..5")
		return
	}
	n := 1
	for _, d := range t.Shape {
		if d <= 0 || n > (1<<26)/d {
			e.fail("invalid tensor dimensions")
			return
		}
		n *= d
	}
	if n != len(t.Data) {
		e.fail("tensor shape/data length mismatch")
		return
	}
	e.ints(len(t.Shape))
	e.ints(t.Shape...)
	e.put(t.Data)
}
func (s *Session) Forward(input *core.Tensor[float32]) (*core.Tensor[float32], error) {
	if s == nil {
		return nil, fmt.Errorf("nativecuda: nil session")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if e := s.ready(); e != nil {
		return nil, e
	}
	var e encoder
	encodeTensor(&e, input)
	if e.err != nil {
		return nil, e.err
	}
	p, err := s.d.forward(e.Bytes())
	if err != nil {
		return nil, err
	}
	r := bytes.NewReader(p)
	var rank int32
	if err = binary.Read(r, binary.LittleEndian, &rank); err != nil || rank < 1 || rank > 5 {
		return nil, fmt.Errorf("nativecuda: invalid output rank")
	}
	shape := make([]int, rank)
	n := 1
	for i := range shape {
		var d int32
		if err = binary.Read(r, binary.LittleEndian, &d); err != nil || d <= 0 || n > (1<<26)/int(d) {
			return nil, fmt.Errorf("nativecuda: invalid output shape")
		}
		shape[i] = int(d)
		n *= int(d)
	}
	v, err := readVector[float32](r)
	if err != nil || len(v) != n || r.Len() != 0 {
		return nil, fmt.Errorf("nativecuda: invalid output data")
	}
	return &core.Tensor[float32]{Shape: shape, Data: v}, nil
}

// TrainMSE performs one native update (one tick for a Stack Step mode).
// Mesh modes on a Grid use ticks, defaulting to one. Flush a Line with further
// calls; pending flights and proxy caches stay in the mounted session.
func (s *Session) TrainMSE(input, target *core.Tensor[float32], mode parallel.TrainMode, lr float64, ticks ...int) (float64, error) {
	if s == nil {
		return 0, fmt.Errorf("nativecuda: nil session")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if e := s.ready(); e != nil {
		return 0, e
	}
	nt := 1
	if len(ticks) > 1 {
		return 0, fmt.Errorf("nativecuda: at most one tick count")
	}
	if len(ticks) == 1 {
		nt = ticks[0]
	}
	if int(mode) < 0 || int(mode) > 33 || nt < 1 || nt > 4096 || math.IsNaN(lr) || math.IsInf(lr, 0) {
		return 0, fmt.Errorf("nativecuda: invalid mode, rate or tick count")
	}
	var e encoder
	encodeTensor(&e, input)
	encodeTensor(&e, target)
	if e.err != nil {
		return 0, e.err
	}
	return s.d.train(int(mode), lr, nt, e.Bytes())
}

// SyncCams applies native CamSync. This does not copy weights into Go.
func (s *Session) SyncCams(when parallel.SyncWhen) error {
	if s == nil {
		return fmt.Errorf("nativecuda: nil session")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if e := s.ready(); e != nil {
		return e
	}
	return s.d.sync(int(when))
}

// StoreSnapshot preserves authoritative native bytes, scale and Float64 biases.
type StoreSnapshot struct {
	DType      core.DType
	Rows, Cols int
	Scale      float32
	F32        []float32
	Native     []byte
	Bias       []float64
}
type ExtraSnapshot struct {
	Kind    int
	Weights []float32
}
type Snapshot struct {
	Stores []StoreSnapshot
	Extras []ExtraSnapshot
}

func readVector[T any](r *bytes.Reader) ([]T, error) {
	var n int32
	var zero T
	size := binary.Size(zero)
	if size <= 0 {
		return nil, fmt.Errorf("invalid vector type")
	}
	if e := binary.Read(r, binary.LittleEndian, &n); e != nil || n < 0 || int64(n) > int64(r.Len()/size) {
		return nil, fmt.Errorf("nativecuda: invalid snapshot vector")
	}
	v := make([]T, n)
	if e := binary.Read(r, binary.LittleEndian, v); e != nil {
		return nil, e
	}
	return v, nil
}
func decodeSnapshot(p []byte) (Snapshot, error) {
	var out Snapshot
	r := bytes.NewReader(p)
	var magic uint32
	var n int32
	if e := binary.Read(r, binary.LittleEndian, &magic); e != nil || magic != 0x3153574e {
		return out, fmt.Errorf("nativecuda: unsupported snapshot")
	}
	if e := binary.Read(r, binary.LittleEndian, &n); e != nil || n < 0 || n > 1<<16 {
		return out, fmt.Errorf("nativecuda: invalid snapshot store count")
	}
	for i := int32(0); i < n; i++ {
		var h struct {
			DT, Rows, Cols int32
			Scale          float32
		}
		if e := binary.Read(r, binary.LittleEndian, &h); e != nil {
			return out, e
		}
		s := StoreSnapshot{DType: core.DType(h.DT), Rows: int(h.Rows), Cols: int(h.Cols), Scale: h.Scale}
		var e error
		if s.F32, e = readVector[float32](r); e != nil {
			return out, e
		}
		if s.Native, e = readVector[byte](r); e != nil {
			return out, e
		}
		if s.Bias, e = readVector[float64](r); e != nil {
			return out, e
		}
		out.Stores = append(out.Stores, s)
	}
	if e := binary.Read(r, binary.LittleEndian, &n); e != nil || n < 0 || n > 4096 {
		return out, fmt.Errorf("nativecuda: invalid extra count")
	}
	for i := int32(0); i < n; i++ {
		var kind int32
		if e := binary.Read(r, binary.LittleEndian, &kind); e != nil {
			return out, e
		}
		v, e := readVector[float32](r)
		if e != nil {
			return out, e
		}
		out.Extras = append(out.Extras, ExtraSnapshot{int(kind), v})
	}
	if r.Len() != 0 {
		return out, fmt.Errorf("nativecuda: trailing snapshot bytes")
	}
	return out, nil
}

// Weights returns an owned snapshot without modifying the original Go model.
func (s *Session) Weights() (Snapshot, error) {
	if s == nil {
		return Snapshot{}, fmt.Errorf("nativecuda: nil session")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if e := s.ready(); e != nil {
		return Snapshot{}, e
	}
	p, e := s.d.export()
	if e != nil {
		return Snapshot{}, e
	}
	return decodeSnapshot(p)
}

// SyncToGo explicitly replaces the original Go stores. All snapshot metadata is
// validated and replacement payloads are prepared before any Go store changes.
// The caller must not concurrently mutate or save that Go model during sync.
func (s *Session) SyncToGo() error {
	if s == nil {
		return fmt.Errorf("nativecuda: nil session")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if e := s.ready(); e != nil {
		return e
	}
	p, e := s.d.export()
	if e != nil {
		return e
	}
	snap, e := decodeSnapshot(p)
	if e != nil {
		return e
	}
	if len(snap.Stores) != len(s.binding.stores) || len(snap.Extras) != len(s.binding.extras) {
		return fmt.Errorf("nativecuda: mount/export binding mismatch")
	}
	fresh := make([]*weights.Store, len(snap.Stores))
	for i, v := range snap.Stores {
		old := s.binding.stores[i]
		if old.Rows != v.Rows || old.Cols != v.Cols || old.DType != v.DType || old.Format != quant.FormatNone || math.IsNaN(float64(v.Scale)) || math.IsInf(float64(v.Scale), 0) {
			return fmt.Errorf("nativecuda: Go store metadata changed since mount")
		}
		init := v.F32
		if v.DType != core.DTypeFloat32 {
			init = nil
		}
		f, err := weights.New(v.Rows, v.Cols, init, v.DType, quant.FormatNone)
		if err != nil {
			return err
		}
		if v.DType == core.DTypeFloat32 {
			if len(v.F32) != v.Rows*v.Cols || len(v.Native) != 0 {
				return fmt.Errorf("invalid Float32 snapshot")
			}
		} else {
			if len(v.Native) != len(f.Native) || len(v.F32) != 0 {
				return fmt.Errorf("invalid native snapshot")
			}
			f.Native = append([]byte(nil), v.Native...)
		}
		f.Scale = v.Scale
		f.Bias = append([]float64(nil), v.Bias...)
		fresh[i] = f
	}
	var apply []func()
	for i, op := range s.binding.extras {
		v := snap.Extras[i]
		switch l := op.(type) {
		case *mamba.Layer:
			off := l.InProj.Weights.Rows*l.InProj.Weights.Cols + l.OutProj.Weights.Rows*l.OutProj.Weights.Cols
			if v.Kind != 18 || len(v.Weights) != off+len(l.ALog)+len(l.DSkip) {
				return fmt.Errorf("invalid Mamba snapshot")
			}
			a := append([]float32(nil), v.Weights[off:off+len(l.ALog)]...)
			d := append([]float32(nil), v.Weights[off+len(l.ALog):]...)
			apply = append(apply, func() { copy(l.ALog, a); copy(l.DSkip, d) })
		case *gdn.Layer:
			if v.Kind != 21 || len(v.Weights) != l.GradWSize() {
				return fmt.Errorf("invalid GDN snapshot")
			}
			off := 0
			blobs := []**quant.Blob{&l.InQKV, &l.InZ, &l.InB, &l.InA, &l.Out}
			newBlobs := make([]*quant.Blob, len(blobs))
			for j, bp := range blobs {
				b := *bp
				n := b.Rows * b.Cols
				nb, err := quant.Pack(quant.FormatNone, v.Weights[off:off+n], b.Rows, b.Cols)
				if err != nil {
					return err
				}
				newBlobs[j] = nb
				off += n
			}
			for j, bp := range blobs {
				bp, nb := bp, newBlobs[j]
				apply = append(apply, func() { *bp = nb })
			}
			for _, dst := range [][]float32{l.ConvWeight, l.ALog, l.DtBias, l.NormGamma} {
				dst := dst
				src := append([]float32(nil), v.Weights[off:off+len(dst)]...)
				off += len(dst)
				apply = append(apply, func() { copy(dst, src) })
			}
		default:
			return fmt.Errorf("unexpected extra binding %T", op)
		}
	}
	for i, f := range fresh {
		*s.binding.stores[i] = *f
	}
	for _, fn := range apply {
		fn()
	}
	return nil
}
func (s *Session) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.d != nil {
		s.d.close()
		s.d = nil
		s.binding = binding{}
	}
	runtime.SetFinalizer(s, nil)
	return nil
}
