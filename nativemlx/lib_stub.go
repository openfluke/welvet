//go:build !cgo || !darwin || !arm64

package nativemlx

import "fmt"

var errNoMLX = fmt.Errorf("nativemlx: libwelvet_mlx not available (need cgo + darwin/arm64 + shipped dylib + Metal GPU)")

func Available() bool { return false }

func Version() string { return "" }

func HasMLX() bool { return false }

func LayerSupported(layer, dtype string) bool { return false }

func SupportedLayers() []string { return nil }

func Detect() (mlxFound, metalOK, gpuOK bool, mlxVer, deviceName string, err error) {
	return false, false, false, "", "", errNoMLX
}

type Session struct{}

func OpenGPU() (*Session, error) { return nil, errNoMLX }

func (s *Session) Close() {}

func (s *Session) Ready() bool { return false }

func (s *Session) DensePrepare(batch, in, out int32, W, bias []float32, dtype string) error {
	return errNoMLX
}
func (s *Session) DensePrepareF32(batch, in, out int32, W, bias []float32) error { return errNoMLX }
func (s *Session) DenseExecuteF32(x, y []float32) error                          { return errNoMLX }
func (s *Session) DenseForward(batch, in, out int32, W, bias, x, y []float32, dtype string) error {
	return errNoMLX
}
func (s *Session) DenseForwardF32(batch, in, out int32, W, bias, x, y []float32) error {
	return errNoMLX
}
func (s *Session) SoftmaxForward(rows, cols int32, x, y []float32, dtype string) error {
	return errNoMLX
}
func (s *Session) SoftmaxForwardF32(rows, cols int32, x, y []float32) error { return errNoMLX }
func (s *Session) RmsNormForward(batch, seq, dim int32, eps float32, gamma, x, y []float32, dtype string) error {
	return errNoMLX
}
func (s *Session) RmsNormForwardF32(batch, seq, dim int32, eps float32, gamma, x, y []float32) error {
	return errNoMLX
}
func (s *Session) LayerNormForward(batch, seq, dim int32, eps float32, gamma, beta, x, y []float32, dtype string) error {
	return errNoMLX
}
func (s *Session) LayerNormForwardF32(batch, seq, dim int32, eps float32, gamma, beta, x, y []float32) error {
	return errNoMLX
}

type Sequential struct{}

func NewSequential() (*Sequential, error) { return nil, errNoMLX }

func (q *Sequential) Free() {}

func (q *Sequential) PushDense(in, out int32, W, bias []float32) error { return errNoMLX }

func (q *Sequential) PushRmsNorm(dim int32, eps float32, gamma []float32) error { return errNoMLX }

func (q *Sequential) PushLayerNorm(dim int32, eps float32, gamma, beta []float32) error {
	return errNoMLX
}

func (q *Sequential) PushSoftmax(cols int32) error { return errNoMLX }

func (q *Sequential) InFeatures() int32  { return 0 }
func (q *Sequential) OutFeatures() int32 { return 0 }
func (q *Sequential) Size() int32        { return 0 }

func (q *Sequential) ForwardF32(s *Session, batch, seqLen int32, x, y []float32) error {
	return errNoMLX
}

func (s *Session) LastPhase() PhaseTiming { return PhaseTiming{} }

func (s *Session) Cnn2Forward(n, h, w, cin, cout int32, W, bias, x, y []float32, dtype string) error {
	return errNoMLX
}
func (s *Session) Cnn2ForwardF32(n, h, w, cin, cout int32, W, bias, x, y []float32) error {
	return errNoMLX
}
func (s *Session) Cnn3Forward(n, d, h, w, cin, cout int32, W, bias, x, y []float32, dtype string) error {
	return errNoMLX
}
func (s *Session) Cnn3ForwardF32(n, d, h, w, cin, cout int32, W, bias, x, y []float32) error {
	return errNoMLX
}
func (s *Session) ConvT2Forward(n, h, w, cin, cout int32, W, bias, x, y []float32, dtype string) error {
	return errNoMLX
}
func (s *Session) ConvT2ForwardF32(n, h, w, cin, cout int32, W, bias, x, y []float32) error {
	return errNoMLX
}
func (s *Session) ConvT3Forward(n, d, h, w, cin, cout int32, W, bias, x, y []float32, dtype string) error {
	return errNoMLX
}
func (s *Session) ConvT3ForwardF32(n, d, h, w, cin, cout int32, W, bias, x, y []float32) error {
	return errNoMLX
}
func (s *Session) LstmForward(seq, batch, inSz, hid int32, wIn, wRec, b, x, y []float32, dtype string) error {
	return errNoMLX
}
func (s *Session) LstmForwardF32(seq, batch, inSz, hid int32, wIn, wRec, b, x, y []float32) error {
	return errNoMLX
}
func (s *Session) EmbeddingForward(vocab, dim, n int32, table []float32, ids []int32, y []float32, dtype string) error {
	return errNoMLX
}
func (s *Session) EmbeddingForwardF32(vocab, dim, n int32, table []float32, ids []int32, y []float32) error {
	return errNoMLX
}
func (s *Session) ResidualForward(n int32, a, b, y []float32, dtype string) error { return errNoMLX }
func (s *Session) ResidualForwardF32(n int32, a, b, y []float32) error             { return errNoMLX }
func (s *Session) SwiGluForward(batch, in, hidden int32, wGate, wUp, wDown, x, y []float32, dtype string) error {
	return errNoMLX
}
func (s *Session) SwiGluForwardF32(batch, in, hidden int32, wGate, wUp, wDown, x, y []float32) error {
	return errNoMLX
}
func (s *Session) MhaForward(batch, seq, dim, heads int32, wQ, wK, wV, wO, x, y []float32, dtype string) error {
	return errNoMLX
}
func (s *Session) MhaForwardF32(batch, seq, dim, heads int32, wQ, wK, wV, wO, x, y []float32) error {
	return errNoMLX
}
