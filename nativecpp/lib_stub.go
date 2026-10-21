//go:build !cgo || !linux || !amd64

package nativecpp

func libLoaded() bool { return false }

var errNoNative = errString("nativecpp: libwelvet not available on this platform (build with cgo on a shipped OS/arch)")

type errString string

func (e errString) Error() string { return string(e) }

const (
	CABBackendCPU  int32 = 0
	CABBackendSIMD int32 = 1
)

func Version() string { return "" }
func GapCount() int   { return 0 }
func PrintGaps()      {}

func DenseForwardF32(batch, in, out, act, dtype, backend int32, wInit []float32, bias []float64, x, pre, post []float32) error {
	return errNoNative
}
func DenseForwardF32CPU(batch, in, out, act, dtype int32, wInit []float32, bias []float64, x, pre, post []float32) error {
	return errNoNative
}
func DenseNormalBPF32(batch, in, out, act, dtype, backend int32, wInit []float32, bias []float64, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (float64, error) {
	return 0, errNoNative
}
func ArchTopology(depth, rows, cols, layersPerCell int32, links []int32, outCoords, outIndices, outRemote []int32) (int32, error) {
	return 0, errNoNative
}
func MHAForwardF32(batch, seq, dModel, numHeads, numKVHeads, headDim, maxSeqLen, mask, pos, dtype, backend int32, Q, K, V, O, x, pre, post []float32) error {
	return errNoNative
}
func MHANormalBPF32(batch, seq, dModel, numHeads, numKVHeads, headDim, maxSeqLen, mask, pos, dtype, backend int32, Q, K, V, O, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (float64, error) {
	return 0, errNoNative
}
func SwiGLUForwardF32(batch, seq, inputDim, intermediateDim, dtype, backend int32, Gate, Up, Down, x, pre, post []float32) error {
	return errNoNative
}
func SwiGLUNormalBPF32(batch, seq, inputDim, intermediateDim, dtype, backend int32, Gate, Up, Down, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (float64, error) {
	return 0, errNoNative
}
func RMSNormForwardF32(batch, seq, dim int32, eps float64, dtype, backend int32, gamma, x, pre, post []float32) error {
	return errNoNative
}
func RMSNormNormalBPF32(batch, seq, dim int32, eps float64, dtype, backend int32, gammaInit, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, gammaAfter []float32) (float64, error) {
	return 0, errNoNative
}
func LayerNormForwardF32(batch, seq, dim int32, eps float64, dtype, backend int32, gamma, beta, x, pre, post []float32) error {
	return errNoNative
}
func LayerNormNormalBPF32(batch, seq, dim int32, eps float64, dtype, backend int32, gammaInit, betaInit, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, gammaAfter, betaAfter []float32) (float64, error) {
	return 0, errNoNative
}
func CNN1ForwardF32(batch, inC, filters, seqLen, kernel, stride, padding, act, dtype, backend int32, W []float32, bias []float64, x, pre, post []float32) error {
	return errNoNative
}
func CNN1NormalBPF32(batch, inC, filters, seqLen, kernel, stride, padding, act, dtype, backend int32, W []float32, bias []float64, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (float64, error) {
	return 0, errNoNative
}
func CNN2ForwardF32(batch, inC, filters, height, width, kernel, stride, padding, act, dtype, backend int32, W []float32, bias []float64, x, pre, post []float32) error {
	return errNoNative
}
func CNN2NormalBPF32(batch, inC, filters, height, width, kernel, stride, padding, act, dtype, backend int32, W []float32, bias []float64, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (float64, error) {
	return 0, errNoNative
}
func CNN3ForwardF32(batch, inC, filters, depth, height, width, kernel, stride, padding, act, dtype, backend int32, W []float32, bias []float64, x, pre, post []float32) error {
	return errNoNative
}
func CNN3NormalBPF32(batch, inC, filters, depth, height, width, kernel, stride, padding, act, dtype, backend int32, W []float32, bias []float64, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (float64, error) {
	return 0, errNoNative
}
func ConvT1ForwardF32(batch, inC, filters, seqLen, kernel, stride, padding, outputPadding, act, dtype, backend int32, W, x, pre, post []float32) error {
	return errNoNative
}
func ConvT2ForwardF32(batch, inC, filters, height, width, kernel, stride, padding, outputPadding, act, dtype, backend int32, W, x, pre, post []float32) error {
	return errNoNative
}
func ConvT3ForwardF32(batch, inC, filters, depth, height, width, kernel, stride, padding, outputPadding, act, dtype, backend int32, W, x, pre, post []float32) error {
	return errNoNative
}
func ConvT1NormalBPF32(batch, inC, filters, seqLen, kernel, stride, padding, outputPadding, act, dtype, backend int32, W, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (float64, error) {
	return 0, errNoNative
}
func ConvT2NormalBPF32(batch, inC, filters, height, width, kernel, stride, padding, outputPadding, act, dtype, backend int32, W, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (float64, error) {
	return 0, errNoNative
}
func ConvT3NormalBPF32(batch, inC, filters, depth, height, width, kernel, stride, padding, outputPadding, act, dtype, backend int32, W, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (float64, error) {
	return 0, errNoNative
}
func RNNForwardF32(batch, inputSize, hiddenSize, seqLen, dtype, backend int32, packed, x, pre, post []float32) error {
	return errNoNative
}
func LSTMForwardF32(batch, inputSize, hiddenSize, seqLen, dtype, backend int32, packed, x, pre, post []float32) error {
	return errNoNative
}
func RNNNormalBPF32(batch, inputSize, hiddenSize, seqLen, dtype, backend int32, packed, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (float64, error) {
	return 0, errNoNative
}
func LSTMNormalBPF32(batch, inputSize, hiddenSize, seqLen, dtype, backend int32, packed, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (float64, error) {
	return 0, errNoNative
}
func EmbeddingForwardF32(batch, vocab, emb, seqLen, inRank, inLast, act, dtype, backend int32, W, x, pre, post []float32) error {
	return errNoNative
}
func EmbeddingNormalBPF32(batch, vocab, emb, seqLen, inRank, inLast, act, dtype, backend int32, wInit, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (float64, error) {
	return 0, errNoNative
}
func SoftmaxForwardF32(n, dim, seqLen, kind, rows, cols int32, temperature, entmaxAlpha float64, gumbelSeed, dtype, backend, rank int32, shape []int32, mask []byte, hier []int32, x, pre, post []float32) error {
	return errNoNative
}
func SoftmaxNormalBPF32(n, dim, seqLen, kind, rows, cols int32, temperature, entmaxAlpha float64, gumbelSeed, dtype, backend, rank int32, shape []int32, mask []byte, hier []int32, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX []float32) (float64, error) {
	return 0, errNoNative
}
func KMeansForwardF32(n, featureDim, numClusters, outputMode, act int32, temperature float64, dtype, backend, rank int32, shape []int32, W, x, pre, post []float32) error {
	return errNoNative
}
func KMeansNormalBPF32(batch, seq, numClusters, featureDim, outputMode, act int32, temperature float64, dtype, backend int32, W, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (float64, error) {
	return 0, errNoNative
}
func SequentialForwardF32(batch, seq, dim, depth, childAct, dtype, backend int32, packed, x, pre, post []float32) error {
	return errNoNative
}
func SequentialNormalBPF32(batch, seq, dim, depth, childAct, dtype, backend int32, packed, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (float64, error) {
	return 0, errNoNative
}
func ResidualForwardF32(batch, seq, dim, depth, childAct, dtype, backend int32, packed, x, pre, post []float32) error {
	return errNoNative
}
func ResidualNormalBPF32(batch, seq, dim, depth, childAct, dtype, backend int32, packed, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (float64, error) {
	return 0, errNoNative
}
func MambaForwardF32(batch, seq, dModel, dState, expand, dtype, backend int32, packed, x, pre, post []float32) error {
	return errNoNative
}
func MambaNormalBPF32(batch, seq, dModel, dState, expand, dtype, backend int32, packed, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (float64, error) {
	return 0, errNoNative
}
func MetacognitionForwardF32(batch, seq, dim, dtype, backend int32, packed, x, pre, post []float32) error {
	return errNoNative
}
func MetacognitionNormalBPF32(batch, seq, dim, dtype, backend int32, packed, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (float64, error) {
	return 0, errNoNative
}
func GDNForwardF32(batch, seq, hidden, numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel, dtype, backend int32, packed, x, pre, post []float32) error {
	return errNoNative
}
func GDNNormalBPF32(batch, seq, hidden, numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel, dtype, backend int32, packed, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (float64, error) {
	return 0, errNoNative
}
func StackNormalBPF32(batch, seq, dim, nChildren, childAct, dtype, backend int32, packed, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (float64, error) {
	return 0, errNoNative
}
func ParallelForwardF32(batch, seq, dim, outFeat, branches, combine, sparseK, dtype, backend int32, disagreeBeta float64, packed, x, pre, post []float32) error {
	return errNoNative
}
func ParallelNormalBPF32(batch, seq, dim, outFeat, branches, combine, sparseK, dtype, backend int32, disagreeBeta float64, packed, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (float64, error) {
	return 0, errNoNative
}
func StackStepBPDenseF32(nChildren, nTicks, batch int32, lr float64, dtype int32, inH, outH, act []int32, packedW, xTicks, targetTicks, wAfter []float32) (float64, error) {
	return 0, errNoNative
}
func SeqMixKindName(kind int32) (string, error) { return "", errNoNative }
func SeqMixMHAKind() int32                       { return 0 }
func SeqMixMambaKind() int32                     { return 0 }
func SeqMixGDNKind() int32                       { return 0 }
