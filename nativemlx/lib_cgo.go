//go:build cgo && darwin && arm64

package nativemlx

/*
#cgo CFLAGS: -I${SRCDIR}/../native/mlx/include
#cgo LDFLAGS: -L${SRCDIR}/../native/mlx/darwin_arm64 -lwelvet_mlx -Wl,-rpath,${SRCDIR}/../native/mlx/darwin_arm64
#include "welvet_mlx_c.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"sync"
	"unsafe"
)

var (
	availOnce sync.Once
	availOK   bool
)

func Available() bool {
	availOnce.Do(func() {
		if C.wmlx_has_mlx() == 0 {
			availOK = false
			return
		}
		mf, mok, gok, _, _, err := Detect()
		availOK = err == nil && mf && mok && gok
	})
	return availOK
}

func Version() string { return C.GoString(C.wmlx_version()) }

func HasMLX() bool { return C.wmlx_has_mlx() != 0 }

func LayerSupported(layer, dtype string) bool {
	cl := C.CString(layer)
	cd := C.CString(dtype)
	defer C.free(unsafe.Pointer(cl))
	defer C.free(unsafe.Pointer(cd))
	return C.wmlx_layer_supported(cl, cd) != 0
}

func SupportedLayers() []string {
	p := C.wmlx_supported_layers()
	if p == nil {
		return nil
	}
	base := uintptr(unsafe.Pointer(p))
	stride := unsafe.Sizeof(uintptr(0))
	var out []string
	for i := uintptr(0); ; i++ {
		elem := *(**C.char)(unsafe.Pointer(base + i*stride))
		if elem == nil {
			break
		}
		out = append(out, C.GoString(elem))
	}
	return out
}

func check(st C.wmlx_status, name string) error {
	if st.code != 0 {
		return fmt.Errorf("nativemlx %s: code=%d %s", name, int(st.code), C.GoString(&st.message[0]))
	}
	return nil
}

func Detect() (mlxFound, metalOK, gpuOK bool, mlxVer, deviceName string, err error) {
	var mf, mok, gok C.int32_t
	var ver [128]C.char
	var dev [256]C.char
	st := C.wmlx_detect(&mf, &mok, &gok, &ver[0], 128, &dev[0], 256)
	if err = check(st, "detect"); err != nil {
		return
	}
	return mf != 0, mok != 0, gok != 0, C.GoString(&ver[0]), C.GoString(&dev[0]), nil
}

func f32Ptr(s []float32) *C.float {
	if len(s) == 0 {
		return nil
	}
	return (*C.float)(unsafe.Pointer(&s[0]))
}

type Session struct {
	p *C.wmlx_session
}

func OpenGPU() (*Session, error) {
	var p *C.wmlx_session
	st := C.wmlx_session_open_gpu(&p)
	if err := check(st, "session_open_gpu"); err != nil {
		return nil, err
	}
	return &Session{p: p}, nil
}

func (s *Session) Close() {
	if s != nil && s.p != nil {
		C.wmlx_session_close(s.p)
		s.p = nil
	}
}

func (s *Session) Ready() bool {
	return s != nil && s.p != nil && C.wmlx_session_ready(s.p) != 0
}

func cstr(s string) *C.char {
	if s == "" {
		s = "float32"
	}
	return C.CString(s)
}

func (s *Session) DensePrepare(batch, in, out int32, W, bias []float32, dtype string) error {
	if s == nil || s.p == nil {
		return fmt.Errorf("nativemlx: nil session")
	}
	if dtype == "" {
		dtype = "float32"
	}
	if !LayerSupported("dense", dtype) {
		return fmt.Errorf("nativemlx: dense/%s not supported on MLX GPU", dtype)
	}
	cd := cstr(dtype)
	defer C.free(unsafe.Pointer(cd))
	st := C.wmlx_dense_prepare_f32(s.p, C.int32_t(batch), C.int32_t(in), C.int32_t(out),
		f32Ptr(W), f32Ptr(bias), cd)
	return check(st, "dense_prepare")
}

func (s *Session) DensePrepareF32(batch, in, out int32, W, bias []float32) error {
	return s.DensePrepare(batch, in, out, W, bias, "float32")
}

func (s *Session) DenseExecuteF32(x, y []float32) error {
	if s == nil || s.p == nil {
		return fmt.Errorf("nativemlx: nil session")
	}
	st := C.wmlx_dense_execute_f32(s.p, f32Ptr(x), f32Ptr(y))
	return check(st, "dense_execute")
}

func (s *Session) DenseForward(batch, in, out int32, W, bias, x, y []float32, dtype string) error {
	if s == nil || s.p == nil {
		return fmt.Errorf("nativemlx: nil session")
	}
	if dtype == "" {
		dtype = "float32"
	}
	if !LayerSupported("dense", dtype) {
		return fmt.Errorf("nativemlx: dense/%s not supported on MLX GPU", dtype)
	}
	cd := cstr(dtype)
	defer C.free(unsafe.Pointer(cd))
	st := C.wmlx_dense_forward_f32(s.p, C.int32_t(batch), C.int32_t(in), C.int32_t(out),
		f32Ptr(W), f32Ptr(bias), f32Ptr(x), f32Ptr(y), cd)
	return check(st, "dense_forward")
}

func (s *Session) DenseForwardF32(batch, in, out int32, W, bias, x, y []float32) error {
	return s.DenseForward(batch, in, out, W, bias, x, y, "float32")
}

func (s *Session) SoftmaxForward(rows, cols int32, x, y []float32, dtype string) error {
	if s == nil || s.p == nil {
		return fmt.Errorf("nativemlx: nil session")
	}
	if dtype == "" {
		dtype = "float32"
	}
	if !LayerSupported("softmax", dtype) {
		return fmt.Errorf("nativemlx: softmax/%s not supported", dtype)
	}
	cd := cstr(dtype)
	defer C.free(unsafe.Pointer(cd))
	st := C.wmlx_softmax_forward_f32(s.p, C.int32_t(rows), C.int32_t(cols), f32Ptr(x), f32Ptr(y), cd)
	return check(st, "softmax_forward")
}

func (s *Session) SoftmaxForwardF32(rows, cols int32, x, y []float32) error {
	return s.SoftmaxForward(rows, cols, x, y, "float32")
}

func (s *Session) RmsNormForward(batch, seq, dim int32, eps float32, gamma, x, y []float32, dtype string) error {
	if s == nil || s.p == nil {
		return fmt.Errorf("nativemlx: nil session")
	}
	if dtype == "" {
		dtype = "float32"
	}
	if !LayerSupported("rmsnorm", dtype) {
		return fmt.Errorf("nativemlx: rmsnorm/%s not supported", dtype)
	}
	cd := cstr(dtype)
	defer C.free(unsafe.Pointer(cd))
	st := C.wmlx_rmsnorm_forward_f32(s.p, C.int32_t(batch), C.int32_t(seq), C.int32_t(dim),
		C.float(eps), f32Ptr(gamma), f32Ptr(x), f32Ptr(y), cd)
	return check(st, "rmsnorm_forward")
}

func (s *Session) RmsNormForwardF32(batch, seq, dim int32, eps float32, gamma, x, y []float32) error {
	return s.RmsNormForward(batch, seq, dim, eps, gamma, x, y, "float32")
}

func (s *Session) LayerNormForward(batch, seq, dim int32, eps float32, gamma, beta, x, y []float32, dtype string) error {
	if s == nil || s.p == nil {
		return fmt.Errorf("nativemlx: nil session")
	}
	if dtype == "" {
		dtype = "float32"
	}
	if !LayerSupported("layernorm", dtype) {
		return fmt.Errorf("nativemlx: layernorm/%s not supported", dtype)
	}
	cd := cstr(dtype)
	defer C.free(unsafe.Pointer(cd))
	st := C.wmlx_layernorm_forward_f32(s.p, C.int32_t(batch), C.int32_t(seq), C.int32_t(dim),
		C.float(eps), f32Ptr(gamma), f32Ptr(beta), f32Ptr(x), f32Ptr(y), cd)
	return check(st, "layernorm_forward")
}

func (s *Session) LayerNormForwardF32(batch, seq, dim int32, eps float32, gamma, beta, x, y []float32) error {
	return s.LayerNormForward(batch, seq, dim, eps, gamma, beta, x, y, "float32")
}

type Sequential struct {
	p *C.wmlx_sequential
}

func NewSequential() (*Sequential, error) {
	var p *C.wmlx_sequential
	st := C.wmlx_seq_create(&p)
	if err := check(st, "seq_create"); err != nil {
		return nil, err
	}
	return &Sequential{p: p}, nil
}

func (q *Sequential) Free() {
	if q != nil && q.p != nil {
		C.wmlx_seq_free(q.p)
		q.p = nil
	}
}

func (q *Sequential) PushDense(in, out int32, W, bias []float32) error {
	if q == nil || q.p == nil {
		return fmt.Errorf("nativemlx: nil seq")
	}
	if !LayerSupported("dense", "float32") {
		return fmt.Errorf("nativemlx: reject push dense — not supported on MLX GPU")
	}
	st := C.wmlx_seq_push_dense(q.p, C.int32_t(in), C.int32_t(out), f32Ptr(W), f32Ptr(bias))
	return check(st, "seq_push_dense")
}

func (q *Sequential) PushRmsNorm(dim int32, eps float32, gamma []float32) error {
	if q == nil || q.p == nil {
		return fmt.Errorf("nativemlx: nil seq")
	}
	if !LayerSupported("rmsnorm", "float32") {
		return fmt.Errorf("nativemlx: reject push rmsnorm")
	}
	st := C.wmlx_seq_push_rmsnorm(q.p, C.int32_t(dim), C.float(eps), f32Ptr(gamma))
	return check(st, "seq_push_rmsnorm")
}

func (q *Sequential) PushLayerNorm(dim int32, eps float32, gamma, beta []float32) error {
	if q == nil || q.p == nil {
		return fmt.Errorf("nativemlx: nil seq")
	}
	if !LayerSupported("layernorm", "float32") {
		return fmt.Errorf("nativemlx: reject push layernorm")
	}
	st := C.wmlx_seq_push_layernorm(q.p, C.int32_t(dim), C.float(eps), f32Ptr(gamma), f32Ptr(beta))
	return check(st, "seq_push_layernorm")
}

func (q *Sequential) PushSoftmax(cols int32) error {
	if q == nil || q.p == nil {
		return fmt.Errorf("nativemlx: nil seq")
	}
	if !LayerSupported("softmax", "float32") {
		return fmt.Errorf("nativemlx: reject push softmax")
	}
	st := C.wmlx_seq_push_softmax(q.p, C.int32_t(cols))
	return check(st, "seq_push_softmax")
}

func (q *Sequential) InFeatures() int32 {
	if q == nil || q.p == nil {
		return 0
	}
	return int32(C.wmlx_seq_in_features(q.p))
}
func (q *Sequential) OutFeatures() int32 {
	if q == nil || q.p == nil {
		return 0
	}
	return int32(C.wmlx_seq_out_features(q.p))
}
func (q *Sequential) Size() int32 {
	if q == nil || q.p == nil {
		return 0
	}
	return int32(C.wmlx_seq_size(q.p))
}

func (q *Sequential) ForwardF32(s *Session, batch, seqLen int32, x, y []float32) error {
	if q == nil || q.p == nil || s == nil || s.p == nil {
		return fmt.Errorf("nativemlx: nil seq/session")
	}
	st := C.wmlx_seq_forward_f32(s.p, q.p, C.int32_t(batch), C.int32_t(seqLen), f32Ptr(x), f32Ptr(y))
	return check(st, "seq_forward")
}

func (s *Session) LastPhase() PhaseTiming {
	var c, f, e C.int64_t
	if s != nil && s.p != nil {
		C.wmlx_last_phase_ns(s.p, &c, &f, &e)
	}
	return PhaseTiming{CreateNs: int64(c), FinalizeNs: int64(f), ExecuteNs: int64(e)}
}

func i32Ptr(v []int32) *C.int32_t {
	if len(v) == 0 {
		return nil
	}
	return (*C.int32_t)(unsafe.Pointer(&v[0]))
}

func (s *Session) Cnn2Forward(n, h, w, cin, cout int32, W, bias, x, y []float32, dtype string) error {
	if s == nil || s.p == nil {
		return fmt.Errorf("nativemlx: nil session")
	}
	if dtype == "" {
		dtype = "float32"
	}
	cd := cstr(dtype)
	defer C.free(unsafe.Pointer(cd))
	st := C.wmlx_cnn2_forward_f32(s.p, C.int32_t(n), C.int32_t(h), C.int32_t(w), C.int32_t(cin),
		C.int32_t(cout), f32Ptr(W), f32Ptr(bias), f32Ptr(x), f32Ptr(y), cd)
	return check(st, "cnn2")
}
func (s *Session) Cnn2ForwardF32(n, h, w, cin, cout int32, W, bias, x, y []float32) error {
	return s.Cnn2Forward(n, h, w, cin, cout, W, bias, x, y, "float32")
}

func (s *Session) Cnn3Forward(n, d, h, w, cin, cout int32, W, bias, x, y []float32, dtype string) error {
	if s == nil || s.p == nil {
		return fmt.Errorf("nativemlx: nil session")
	}
	if dtype == "" {
		dtype = "float32"
	}
	cd := cstr(dtype)
	defer C.free(unsafe.Pointer(cd))
	st := C.wmlx_cnn3_forward_f32(s.p, C.int32_t(n), C.int32_t(d), C.int32_t(h), C.int32_t(w),
		C.int32_t(cin), C.int32_t(cout), f32Ptr(W), f32Ptr(bias), f32Ptr(x), f32Ptr(y), cd)
	return check(st, "cnn3")
}
func (s *Session) Cnn3ForwardF32(n, d, h, w, cin, cout int32, W, bias, x, y []float32) error {
	return s.Cnn3Forward(n, d, h, w, cin, cout, W, bias, x, y, "float32")
}

func (s *Session) ConvT2Forward(n, h, w, cin, cout int32, W, bias, x, y []float32, dtype string) error {
	if s == nil || s.p == nil {
		return fmt.Errorf("nativemlx: nil session")
	}
	if dtype == "" {
		dtype = "float32"
	}
	cd := cstr(dtype)
	defer C.free(unsafe.Pointer(cd))
	st := C.wmlx_convt2_forward_f32(s.p, C.int32_t(n), C.int32_t(h), C.int32_t(w), C.int32_t(cin),
		C.int32_t(cout), f32Ptr(W), f32Ptr(bias), f32Ptr(x), f32Ptr(y), cd)
	return check(st, "convt2")
}
func (s *Session) ConvT2ForwardF32(n, h, w, cin, cout int32, W, bias, x, y []float32) error {
	return s.ConvT2Forward(n, h, w, cin, cout, W, bias, x, y, "float32")
}

func (s *Session) ConvT3Forward(n, d, h, w, cin, cout int32, W, bias, x, y []float32, dtype string) error {
	if s == nil || s.p == nil {
		return fmt.Errorf("nativemlx: nil session")
	}
	if dtype == "" {
		dtype = "float32"
	}
	cd := cstr(dtype)
	defer C.free(unsafe.Pointer(cd))
	st := C.wmlx_convt3_forward_f32(s.p, C.int32_t(n), C.int32_t(d), C.int32_t(h), C.int32_t(w),
		C.int32_t(cin), C.int32_t(cout), f32Ptr(W), f32Ptr(bias), f32Ptr(x), f32Ptr(y), cd)
	return check(st, "convt3")
}
func (s *Session) ConvT3ForwardF32(n, d, h, w, cin, cout int32, W, bias, x, y []float32) error {
	return s.ConvT3Forward(n, d, h, w, cin, cout, W, bias, x, y, "float32")
}

func (s *Session) LstmForward(seq, batch, inSz, hid int32, wIn, wRec, b, x, y []float32, dtype string) error {
	if s == nil || s.p == nil {
		return fmt.Errorf("nativemlx: nil session")
	}
	if dtype == "" {
		dtype = "float32"
	}
	cd := cstr(dtype)
	defer C.free(unsafe.Pointer(cd))
	st := C.wmlx_lstm_forward_f32(s.p, C.int32_t(seq), C.int32_t(batch), C.int32_t(inSz),
		C.int32_t(hid), f32Ptr(wIn), f32Ptr(wRec), f32Ptr(b), f32Ptr(x), f32Ptr(y), cd)
	return check(st, "lstm")
}
func (s *Session) LstmForwardF32(seq, batch, inSz, hid int32, wIn, wRec, b, x, y []float32) error {
	return s.LstmForward(seq, batch, inSz, hid, wIn, wRec, b, x, y, "float32")
}

func (s *Session) EmbeddingForward(vocab, dim, n int32, table []float32, ids []int32, y []float32, dtype string) error {
	if s == nil || s.p == nil {
		return fmt.Errorf("nativemlx: nil session")
	}
	if dtype == "" {
		dtype = "float32"
	}
	cd := cstr(dtype)
	defer C.free(unsafe.Pointer(cd))
	st := C.wmlx_embedding_forward_f32(s.p, C.int32_t(vocab), C.int32_t(dim), C.int32_t(n),
		f32Ptr(table), i32Ptr(ids), f32Ptr(y), cd)
	return check(st, "embedding")
}
func (s *Session) EmbeddingForwardF32(vocab, dim, n int32, table []float32, ids []int32, y []float32) error {
	return s.EmbeddingForward(vocab, dim, n, table, ids, y, "float32")
}

func (s *Session) ResidualForward(n int32, a, b, y []float32, dtype string) error {
	if s == nil || s.p == nil {
		return fmt.Errorf("nativemlx: nil session")
	}
	if dtype == "" {
		dtype = "float32"
	}
	cd := cstr(dtype)
	defer C.free(unsafe.Pointer(cd))
	st := C.wmlx_residual_forward_f32(s.p, C.int32_t(n), f32Ptr(a), f32Ptr(b), f32Ptr(y), cd)
	return check(st, "residual")
}
func (s *Session) ResidualForwardF32(n int32, a, b, y []float32) error {
	return s.ResidualForward(n, a, b, y, "float32")
}

func (s *Session) SwiGluForward(batch, in, hidden int32, wGate, wUp, wDown, x, y []float32, dtype string) error {
	if s == nil || s.p == nil {
		return fmt.Errorf("nativemlx: nil session")
	}
	if dtype == "" {
		dtype = "float32"
	}
	cd := cstr(dtype)
	defer C.free(unsafe.Pointer(cd))
	st := C.wmlx_swiglu_forward_f32(s.p, C.int32_t(batch), C.int32_t(in), C.int32_t(hidden),
		f32Ptr(wGate), f32Ptr(wUp), f32Ptr(wDown), f32Ptr(x), f32Ptr(y), cd)
	return check(st, "swiglu")
}
func (s *Session) SwiGluForwardF32(batch, in, hidden int32, wGate, wUp, wDown, x, y []float32) error {
	return s.SwiGluForward(batch, in, hidden, wGate, wUp, wDown, x, y, "float32")
}

func (s *Session) MhaForward(batch, seq, dim, heads int32, wQ, wK, wV, wO, x, y []float32, dtype string) error {
	if s == nil || s.p == nil {
		return fmt.Errorf("nativemlx: nil session")
	}
	if dtype == "" {
		dtype = "float32"
	}
	cd := cstr(dtype)
	defer C.free(unsafe.Pointer(cd))
	st := C.wmlx_mha_forward_f32(s.p, C.int32_t(batch), C.int32_t(seq), C.int32_t(dim),
		C.int32_t(heads), f32Ptr(wQ), f32Ptr(wK), f32Ptr(wV), f32Ptr(wO), f32Ptr(x), f32Ptr(y), cd)
	return check(st, "mha")
}
func (s *Session) MhaForwardF32(batch, seq, dim, heads int32, wQ, wK, wV, wO, x, y []float32) error {
	return s.MhaForward(batch, seq, dim, heads, wQ, wK, wV, wO, x, y, "float32")
}
