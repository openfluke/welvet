//go:build !welvet_metal && ((cgo && linux && (amd64 || arm64)) || (cgo && darwin && arm64) || (cgo && windows && (amd64 || arm64)))

package nativecpp

/*
#cgo CFLAGS: -I${SRCDIR}/../native/cpp/include
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/../native/cpp/linux_amd64 -lwelvet -Wl,-rpath,${SRCDIR}/../native/cpp/linux_amd64
#cgo linux,arm64 LDFLAGS: -L${SRCDIR}/../native/cpp/linux_arm64 -lwelvet -Wl,-rpath,${SRCDIR}/../native/cpp/linux_arm64
#cgo darwin,arm64 LDFLAGS: -L${SRCDIR}/../native/cpp/darwin_arm64 -lwelvet -Wl,-rpath,${SRCDIR}/../native/cpp/darwin_arm64
#cgo windows,amd64 LDFLAGS: -L${SRCDIR}/../native/cpp/windows_amd64 -lwelvet
#cgo windows,arm64 LDFLAGS: -L${SRCDIR}/../native/cpp/windows_arm64 -lwelvet
#include "welvet_c.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

func libLoaded() bool { return true }

func checkStatus(st C.welvet_status, name string) error {
	if st.code != 0 {
		return fmt.Errorf("nativecpp %s: code=%d %s", name, int(st.code), C.GoString(&st.message[0]))
	}
	recordCompleted(name)
	return nil
}

func f32Ptr(s []float32) *C.float {
	if len(s) == 0 {
		return nil
	}
	return (*C.float)(unsafe.Pointer(&s[0]))
}

func f64Ptr(s []float64) *C.double {
	if len(s) == 0 {
		return nil
	}
	return (*C.double)(unsafe.Pointer(&s[0]))
}

func i32Ptr(s []int32) *C.int32_t {
	if len(s) == 0 {
		return nil
	}
	return (*C.int32_t)(unsafe.Pointer(&s[0]))
}

func u8Ptr(s []byte) *C.uint8_t {
	if len(s) == 0 {
		return nil
	}
	return (*C.uint8_t)(unsafe.Pointer(&s[0]))
}

// Version returns welvet.cpp welvet_version().
func Version() string {
	return C.GoString(C.welvet_version())
}

// GapCount returns welvet_gap_count().
func GapCount() int {
	return int(C.welvet_gap_count())
}

// PrintGaps calls welvet_print_gaps().
func PrintGaps() { C.welvet_print_gaps() }

// CABBackendCPU / CABBackendSIMD match C ABI backend ints (exact_bit Go CPU = 0).
const (
	CABBackendCPU  int32 = 0
	CABBackendSIMD int32 = 1
)

// DenseForwardF32 calls welvet_dense_forward_f32 (backend 0 = exact Go CPU path).
func DenseForwardF32(batch, in, out, act, dtype, backend int32, wInit []float32, bias []float64, x, pre, post []float32) error {
	needW := int(out * in)
	needX := int(batch * in)
	needY := int(batch * out)
	if len(wInit) < needW || len(x) < needX || len(pre) < needY || len(post) < needY {
		return fmt.Errorf("nativecpp: dense_forward buffer length mismatch")
	}
	st := C.welvet_dense_forward_f32(
		C.int32_t(batch), C.int32_t(in), C.int32_t(out), C.int32_t(act), C.int32_t(dtype), C.int32_t(backend),
		f32Ptr(wInit), f64Ptr(bias), C.size_t(len(bias)),
		f32Ptr(x), f32Ptr(pre), f32Ptr(post),
	)
	return checkStatus(st, "dense_forward")
}

// DenseForwardF32CPU is the back-compat CPU wrapper.
func DenseForwardF32CPU(batch, in, out, act, dtype int32, wInit []float32, bias []float64, x, pre, post []float32) error {
	return DenseForwardF32(batch, in, out, act, dtype, CABBackendCPU, wInit, bias, x, pre, post)
}

// DenseNormalBPF32 calls welvet_dense_normalbp_f32.
// useTarget: false = gradOrTarget is gradOut; true = MSE target. lr==0 skips SGD.
func DenseNormalBPF32(batch, in, out, act, dtype, backend int32, wInit []float32, bias []float64, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (loss float64, err error) {
	needW := int(out * in)
	needX := int(batch * in)
	needY := int(batch * out)
	if len(wInit) < needW || len(x) < needX || len(gradOrTarget) < needY ||
		len(pre) < needY || len(post) < needY || len(dX) < needX || len(dW) < needW || len(wAfter) < needW {
		return 0, fmt.Errorf("nativecpp: dense_normalbp buffer length mismatch")
	}
	ut := C.int32_t(0)
	if useTarget {
		ut = 1
	}
	var closs C.double
	st := C.welvet_dense_normalbp_f32(
		C.int32_t(batch), C.int32_t(in), C.int32_t(out), C.int32_t(act), C.int32_t(dtype), C.int32_t(backend),
		f32Ptr(wInit), f64Ptr(bias), C.size_t(len(bias)), f32Ptr(x),
		f32Ptr(gradOrTarget), ut, C.double(lr),
		f32Ptr(pre), f32Ptr(post), f32Ptr(dX), f32Ptr(dW), f32Ptr(wAfter), &closs,
	)
	return float64(closs), checkStatus(st, "dense_normalbp")
}

// ArchTopology calls welvet_arch_topology.
func ArchTopology(depth, rows, cols, layersPerCell int32, links []int32, outCoords, outIndices, outRemote []int32) (nCells int32, err error) {
	var n C.int32_t
	st := C.welvet_arch_topology(
		C.int32_t(depth), C.int32_t(rows), C.int32_t(cols), C.int32_t(layersPerCell),
		i32Ptr(links), C.int32_t(len(links)/8),
		&n, i32Ptr(outCoords), i32Ptr(outIndices), i32Ptr(outRemote),
	)
	return int32(n), checkStatus(st, "arch_topology")
}

func MHAForwardF32(batch, seq, dModel, numHeads, numKVHeads, headDim, maxSeqLen, mask, pos, dtype, backend int32, Q, K, V, O, x, pre, post []float32) error {
	st := C.welvet_mha_forward_f32(
		C.int32_t(batch), C.int32_t(seq), C.int32_t(dModel), C.int32_t(numHeads), C.int32_t(numKVHeads),
		C.int32_t(headDim), C.int32_t(maxSeqLen), C.int32_t(mask), C.int32_t(pos), C.int32_t(dtype), C.int32_t(backend),
		f32Ptr(Q), f32Ptr(K), f32Ptr(V), f32Ptr(O), f32Ptr(x), f32Ptr(pre), f32Ptr(post),
	)
	return checkStatus(st, "mha_forward")
}

func MHANormalBPF32(batch, seq, dModel, numHeads, numKVHeads, headDim, maxSeqLen, mask, pos, dtype, backend int32, Q, K, V, O, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (loss float64, err error) {
	ut := C.int32_t(0)
	if useTarget {
		ut = 1
	}
	var closs C.double
	st := C.welvet_mha_normalbp_f32(
		C.int32_t(batch), C.int32_t(seq), C.int32_t(dModel), C.int32_t(numHeads), C.int32_t(numKVHeads),
		C.int32_t(headDim), C.int32_t(maxSeqLen), C.int32_t(mask), C.int32_t(pos), C.int32_t(dtype), C.int32_t(backend),
		f32Ptr(Q), f32Ptr(K), f32Ptr(V), f32Ptr(O), f32Ptr(x), f32Ptr(gradOrTarget), ut, C.double(lr),
		f32Ptr(pre), f32Ptr(post), f32Ptr(dX), f32Ptr(dW), f32Ptr(wAfter), &closs,
	)
	return float64(closs), checkStatus(st, "mha_normalbp")
}

func SwiGLUForwardF32(batch, seq, inputDim, intermediateDim, dtype, backend int32, Gate, Up, Down, x, pre, post []float32) error {
	st := C.welvet_swiglu_forward_f32(
		C.int32_t(batch), C.int32_t(seq), C.int32_t(inputDim), C.int32_t(intermediateDim),
		C.int32_t(dtype), C.int32_t(backend),
		f32Ptr(Gate), f32Ptr(Up), f32Ptr(Down), f32Ptr(x), f32Ptr(pre), f32Ptr(post),
	)
	return checkStatus(st, "swiglu_forward")
}

func SwiGLUNormalBPF32(batch, seq, inputDim, intermediateDim, dtype, backend int32, Gate, Up, Down, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (loss float64, err error) {
	ut := C.int32_t(0)
	if useTarget {
		ut = 1
	}
	var closs C.double
	st := C.welvet_swiglu_normalbp_f32(
		C.int32_t(batch), C.int32_t(seq), C.int32_t(inputDim), C.int32_t(intermediateDim),
		C.int32_t(dtype), C.int32_t(backend),
		f32Ptr(Gate), f32Ptr(Up), f32Ptr(Down), f32Ptr(x), f32Ptr(gradOrTarget), ut, C.double(lr),
		f32Ptr(pre), f32Ptr(post), f32Ptr(dX), f32Ptr(dW), f32Ptr(wAfter), &closs,
	)
	return float64(closs), checkStatus(st, "swiglu_normalbp")
}

func RMSNormForwardF32(batch, seq, dim int32, eps float64, dtype, backend int32, gamma, x, pre, post []float32) error {
	st := C.welvet_rmsnorm_forward_f32(
		C.int32_t(batch), C.int32_t(seq), C.int32_t(dim), C.double(eps), C.int32_t(dtype), C.int32_t(backend),
		f32Ptr(gamma), f32Ptr(x), f32Ptr(pre), f32Ptr(post),
	)
	return checkStatus(st, "rmsnorm_forward")
}

func RMSNormNormalBPF32(batch, seq, dim int32, eps float64, dtype, backend int32, gammaInit, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, gammaAfter []float32) (loss float64, err error) {
	ut := C.int32_t(0)
	if useTarget {
		ut = 1
	}
	var closs C.double
	st := C.welvet_rmsnorm_normalbp_f32(
		C.int32_t(batch), C.int32_t(seq), C.int32_t(dim), C.double(eps), C.int32_t(dtype), C.int32_t(backend),
		f32Ptr(gammaInit), f32Ptr(x), f32Ptr(gradOrTarget), ut, C.double(lr),
		f32Ptr(pre), f32Ptr(post), f32Ptr(dX), f32Ptr(dW), f32Ptr(gammaAfter), &closs,
	)
	return float64(closs), checkStatus(st, "rmsnorm_normalbp")
}

func LayerNormForwardF32(batch, seq, dim int32, eps float64, dtype, backend int32, gamma, beta, x, pre, post []float32) error {
	st := C.welvet_layernorm_forward_f32(
		C.int32_t(batch), C.int32_t(seq), C.int32_t(dim), C.double(eps), C.int32_t(dtype), C.int32_t(backend),
		f32Ptr(gamma), f32Ptr(beta), f32Ptr(x), f32Ptr(pre), f32Ptr(post),
	)
	return checkStatus(st, "layernorm_forward")
}

func LayerNormNormalBPF32(batch, seq, dim int32, eps float64, dtype, backend int32, gammaInit, betaInit, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, gammaAfter, betaAfter []float32) (loss float64, err error) {
	ut := C.int32_t(0)
	if useTarget {
		ut = 1
	}
	var closs C.double
	st := C.welvet_layernorm_normalbp_f32(
		C.int32_t(batch), C.int32_t(seq), C.int32_t(dim), C.double(eps), C.int32_t(dtype), C.int32_t(backend),
		f32Ptr(gammaInit), f32Ptr(betaInit), f32Ptr(x), f32Ptr(gradOrTarget), ut, C.double(lr),
		f32Ptr(pre), f32Ptr(post), f32Ptr(dX), f32Ptr(dW), f32Ptr(gammaAfter), f32Ptr(betaAfter), &closs,
	)
	return float64(closs), checkStatus(st, "layernorm_normalbp")
}

func CNN1ForwardF32(batch, inC, filters, seqLen, kernel, stride, padding, act, dtype, backend int32, W []float32, bias []float64, x, pre, post []float32) error {
	st := C.welvet_cnn1_forward_f32(
		C.int32_t(batch), C.int32_t(inC), C.int32_t(filters), C.int32_t(seqLen), C.int32_t(kernel),
		C.int32_t(stride), C.int32_t(padding), C.int32_t(act), C.int32_t(dtype), C.int32_t(backend),
		f32Ptr(W), f64Ptr(bias), C.size_t(len(bias)), f32Ptr(x), f32Ptr(pre), f32Ptr(post),
	)
	return checkStatus(st, "cnn1_forward")
}

func CNN1NormalBPF32(batch, inC, filters, seqLen, kernel, stride, padding, act, dtype, backend int32, W []float32, bias []float64, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (loss float64, err error) {
	ut := C.int32_t(0)
	if useTarget {
		ut = 1
	}
	var closs C.double
	st := C.welvet_cnn1_normalbp_f32(
		C.int32_t(batch), C.int32_t(inC), C.int32_t(filters), C.int32_t(seqLen), C.int32_t(kernel),
		C.int32_t(stride), C.int32_t(padding), C.int32_t(act), C.int32_t(dtype), C.int32_t(backend),
		f32Ptr(W), f64Ptr(bias), C.size_t(len(bias)), f32Ptr(x), f32Ptr(gradOrTarget), ut, C.double(lr),
		f32Ptr(pre), f32Ptr(post), f32Ptr(dX), f32Ptr(dW), f32Ptr(wAfter), &closs,
	)
	return float64(closs), checkStatus(st, "cnn1_normalbp")
}

func CNN2ForwardF32(batch, inC, filters, height, width, kernel, stride, padding, act, dtype, backend int32, W []float32, bias []float64, x, pre, post []float32) error {
	st := C.welvet_cnn2_forward_f32(
		C.int32_t(batch), C.int32_t(inC), C.int32_t(filters), C.int32_t(height), C.int32_t(width),
		C.int32_t(kernel), C.int32_t(stride), C.int32_t(padding), C.int32_t(act), C.int32_t(dtype), C.int32_t(backend),
		f32Ptr(W), f64Ptr(bias), C.size_t(len(bias)), f32Ptr(x), f32Ptr(pre), f32Ptr(post),
	)
	return checkStatus(st, "cnn2_forward")
}

func CNN2NormalBPF32(batch, inC, filters, height, width, kernel, stride, padding, act, dtype, backend int32, W []float32, bias []float64, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (loss float64, err error) {
	ut := C.int32_t(0)
	if useTarget {
		ut = 1
	}
	var closs C.double
	st := C.welvet_cnn2_normalbp_f32(
		C.int32_t(batch), C.int32_t(inC), C.int32_t(filters), C.int32_t(height), C.int32_t(width),
		C.int32_t(kernel), C.int32_t(stride), C.int32_t(padding), C.int32_t(act), C.int32_t(dtype), C.int32_t(backend),
		f32Ptr(W), f64Ptr(bias), C.size_t(len(bias)), f32Ptr(x), f32Ptr(gradOrTarget), ut, C.double(lr),
		f32Ptr(pre), f32Ptr(post), f32Ptr(dX), f32Ptr(dW), f32Ptr(wAfter), &closs,
	)
	return float64(closs), checkStatus(st, "cnn2_normalbp")
}

func CNN3ForwardF32(batch, inC, filters, depth, height, width, kernel, stride, padding, act, dtype, backend int32, W []float32, bias []float64, x, pre, post []float32) error {
	st := C.welvet_cnn3_forward_f32(
		C.int32_t(batch), C.int32_t(inC), C.int32_t(filters), C.int32_t(depth), C.int32_t(height), C.int32_t(width),
		C.int32_t(kernel), C.int32_t(stride), C.int32_t(padding), C.int32_t(act), C.int32_t(dtype), C.int32_t(backend),
		f32Ptr(W), f64Ptr(bias), C.size_t(len(bias)), f32Ptr(x), f32Ptr(pre), f32Ptr(post),
	)
	return checkStatus(st, "cnn3_forward")
}

func CNN3NormalBPF32(batch, inC, filters, depth, height, width, kernel, stride, padding, act, dtype, backend int32, W []float32, bias []float64, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (loss float64, err error) {
	ut := C.int32_t(0)
	if useTarget {
		ut = 1
	}
	var closs C.double
	st := C.welvet_cnn3_normalbp_f32(
		C.int32_t(batch), C.int32_t(inC), C.int32_t(filters), C.int32_t(depth), C.int32_t(height), C.int32_t(width),
		C.int32_t(kernel), C.int32_t(stride), C.int32_t(padding), C.int32_t(act), C.int32_t(dtype), C.int32_t(backend),
		f32Ptr(W), f64Ptr(bias), C.size_t(len(bias)), f32Ptr(x), f32Ptr(gradOrTarget), ut, C.double(lr),
		f32Ptr(pre), f32Ptr(post), f32Ptr(dX), f32Ptr(dW), f32Ptr(wAfter), &closs,
	)
	return float64(closs), checkStatus(st, "cnn3_normalbp")
}

func ConvT1ForwardF32(batch, inC, filters, seqLen, kernel, stride, padding, outputPadding, act, dtype, backend int32, W, x, pre, post []float32) error {
	st := C.welvet_convt1_forward_f32(
		C.int32_t(batch), C.int32_t(inC), C.int32_t(filters), C.int32_t(seqLen), C.int32_t(kernel),
		C.int32_t(stride), C.int32_t(padding), C.int32_t(outputPadding), C.int32_t(act), C.int32_t(dtype), C.int32_t(backend),
		f32Ptr(W), f32Ptr(x), f32Ptr(pre), f32Ptr(post),
	)
	return checkStatus(st, "convt1_forward")
}

func ConvT2ForwardF32(batch, inC, filters, height, width, kernel, stride, padding, outputPadding, act, dtype, backend int32, W, x, pre, post []float32) error {
	st := C.welvet_convt2_forward_f32(
		C.int32_t(batch), C.int32_t(inC), C.int32_t(filters), C.int32_t(height), C.int32_t(width),
		C.int32_t(kernel), C.int32_t(stride), C.int32_t(padding), C.int32_t(outputPadding), C.int32_t(act), C.int32_t(dtype), C.int32_t(backend),
		f32Ptr(W), f32Ptr(x), f32Ptr(pre), f32Ptr(post),
	)
	return checkStatus(st, "convt2_forward")
}

func ConvT3ForwardF32(batch, inC, filters, depth, height, width, kernel, stride, padding, outputPadding, act, dtype, backend int32, W, x, pre, post []float32) error {
	st := C.welvet_convt3_forward_f32(
		C.int32_t(batch), C.int32_t(inC), C.int32_t(filters), C.int32_t(depth), C.int32_t(height), C.int32_t(width),
		C.int32_t(kernel), C.int32_t(stride), C.int32_t(padding), C.int32_t(outputPadding), C.int32_t(act), C.int32_t(dtype), C.int32_t(backend),
		f32Ptr(W), f32Ptr(x), f32Ptr(pre), f32Ptr(post),
	)
	return checkStatus(st, "convt3_forward")
}

func ConvT1NormalBPF32(batch, inC, filters, seqLen, kernel, stride, padding, outputPadding, act, dtype, backend int32, W, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (loss float64, err error) {
	ut := C.int32_t(0)
	if useTarget {
		ut = 1
	}
	var closs C.double
	st := C.welvet_convt1_normalbp_f32(
		C.int32_t(batch), C.int32_t(inC), C.int32_t(filters), C.int32_t(seqLen), C.int32_t(kernel),
		C.int32_t(stride), C.int32_t(padding), C.int32_t(outputPadding), C.int32_t(act), C.int32_t(dtype), C.int32_t(backend),
		f32Ptr(W), f32Ptr(x), f32Ptr(gradOrTarget), ut, C.double(lr),
		f32Ptr(pre), f32Ptr(post), f32Ptr(dX), f32Ptr(dW), f32Ptr(wAfter), &closs,
	)
	return float64(closs), checkStatus(st, "convt1_normalbp")
}

func ConvT2NormalBPF32(batch, inC, filters, height, width, kernel, stride, padding, outputPadding, act, dtype, backend int32, W, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (loss float64, err error) {
	ut := C.int32_t(0)
	if useTarget {
		ut = 1
	}
	var closs C.double
	st := C.welvet_convt2_normalbp_f32(
		C.int32_t(batch), C.int32_t(inC), C.int32_t(filters), C.int32_t(height), C.int32_t(width),
		C.int32_t(kernel), C.int32_t(stride), C.int32_t(padding), C.int32_t(outputPadding), C.int32_t(act), C.int32_t(dtype), C.int32_t(backend),
		f32Ptr(W), f32Ptr(x), f32Ptr(gradOrTarget), ut, C.double(lr),
		f32Ptr(pre), f32Ptr(post), f32Ptr(dX), f32Ptr(dW), f32Ptr(wAfter), &closs,
	)
	return float64(closs), checkStatus(st, "convt2_normalbp")
}

func ConvT3NormalBPF32(batch, inC, filters, depth, height, width, kernel, stride, padding, outputPadding, act, dtype, backend int32, W, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (loss float64, err error) {
	ut := C.int32_t(0)
	if useTarget {
		ut = 1
	}
	var closs C.double
	st := C.welvet_convt3_normalbp_f32(
		C.int32_t(batch), C.int32_t(inC), C.int32_t(filters), C.int32_t(depth), C.int32_t(height), C.int32_t(width),
		C.int32_t(kernel), C.int32_t(stride), C.int32_t(padding), C.int32_t(outputPadding), C.int32_t(act), C.int32_t(dtype), C.int32_t(backend),
		f32Ptr(W), f32Ptr(x), f32Ptr(gradOrTarget), ut, C.double(lr),
		f32Ptr(pre), f32Ptr(post), f32Ptr(dX), f32Ptr(dW), f32Ptr(wAfter), &closs,
	)
	return float64(closs), checkStatus(st, "convt3_normalbp")
}

func RNNForwardF32(batch, inputSize, hiddenSize, seqLen, dtype, backend int32, packed, x, pre, post []float32) error {
	st := C.welvet_rnn_forward_f32(
		C.int32_t(batch), C.int32_t(inputSize), C.int32_t(hiddenSize), C.int32_t(seqLen),
		C.int32_t(dtype), C.int32_t(backend), f32Ptr(packed), f32Ptr(x), f32Ptr(pre), f32Ptr(post),
	)
	return checkStatus(st, "rnn_forward")
}

func LSTMForwardF32(batch, inputSize, hiddenSize, seqLen, dtype, backend int32, packed, x, pre, post []float32) error {
	st := C.welvet_lstm_forward_f32(
		C.int32_t(batch), C.int32_t(inputSize), C.int32_t(hiddenSize), C.int32_t(seqLen),
		C.int32_t(dtype), C.int32_t(backend), f32Ptr(packed), f32Ptr(x), f32Ptr(pre), f32Ptr(post),
	)
	return checkStatus(st, "lstm_forward")
}

func RNNNormalBPF32(batch, inputSize, hiddenSize, seqLen, dtype, backend int32, packed, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (loss float64, err error) {
	ut := C.int32_t(0)
	if useTarget {
		ut = 1
	}
	var closs C.double
	st := C.welvet_rnn_normalbp_f32(
		C.int32_t(batch), C.int32_t(inputSize), C.int32_t(hiddenSize), C.int32_t(seqLen),
		C.int32_t(dtype), C.int32_t(backend), f32Ptr(packed), f32Ptr(x), f32Ptr(gradOrTarget), ut, C.double(lr),
		f32Ptr(pre), f32Ptr(post), f32Ptr(dX), f32Ptr(dW), f32Ptr(wAfter), &closs,
	)
	return float64(closs), checkStatus(st, "rnn_normalbp")
}

func LSTMNormalBPF32(batch, inputSize, hiddenSize, seqLen, dtype, backend int32, packed, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (loss float64, err error) {
	ut := C.int32_t(0)
	if useTarget {
		ut = 1
	}
	var closs C.double
	st := C.welvet_lstm_normalbp_f32(
		C.int32_t(batch), C.int32_t(inputSize), C.int32_t(hiddenSize), C.int32_t(seqLen),
		C.int32_t(dtype), C.int32_t(backend), f32Ptr(packed), f32Ptr(x), f32Ptr(gradOrTarget), ut, C.double(lr),
		f32Ptr(pre), f32Ptr(post), f32Ptr(dX), f32Ptr(dW), f32Ptr(wAfter), &closs,
	)
	return float64(closs), checkStatus(st, "lstm_normalbp")
}

func EmbeddingForwardF32(batch, vocab, emb, seqLen, inRank, inLast, act, dtype, backend int32, W, x, pre, post []float32) error {
	st := C.welvet_embedding_forward_f32(
		C.int32_t(batch), C.int32_t(vocab), C.int32_t(emb), C.int32_t(seqLen), C.int32_t(inRank), C.int32_t(inLast),
		C.int32_t(act), C.int32_t(dtype), C.int32_t(backend), f32Ptr(W), f32Ptr(x), f32Ptr(pre), f32Ptr(post),
	)
	return checkStatus(st, "embedding_forward")
}

func EmbeddingNormalBPF32(batch, vocab, emb, seqLen, inRank, inLast, act, dtype, backend int32, wInit, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (loss float64, err error) {
	ut := C.int32_t(0)
	if useTarget {
		ut = 1
	}
	var closs C.double
	st := C.welvet_embedding_normalbp_f32(
		C.int32_t(batch), C.int32_t(vocab), C.int32_t(emb), C.int32_t(seqLen), C.int32_t(inRank), C.int32_t(inLast),
		C.int32_t(act), C.int32_t(dtype), C.int32_t(backend), f32Ptr(wInit), f32Ptr(x), f32Ptr(gradOrTarget), ut, C.double(lr),
		f32Ptr(pre), f32Ptr(post), f32Ptr(dX), f32Ptr(dW), f32Ptr(wAfter), &closs,
	)
	return float64(closs), checkStatus(st, "embedding_normalbp")
}

func SoftmaxForwardF32(n, dim, seqLen, kind, rows, cols int32, temperature, entmaxAlpha float64, gumbelSeed, dtype, backend, rank int32, shape []int32, mask []byte, hier []int32, x, pre, post []float32) error {
	st := C.welvet_softmax_forward_f32(
		C.int32_t(n), C.int32_t(dim), C.int32_t(seqLen), C.int32_t(kind), C.int32_t(rows), C.int32_t(cols),
		C.double(temperature), C.double(entmaxAlpha), C.int32_t(gumbelSeed), C.int32_t(dtype), C.int32_t(backend),
		C.int32_t(rank), i32Ptr(shape), u8Ptr(mask), C.int32_t(len(mask)), i32Ptr(hier), C.int32_t(len(hier)),
		f32Ptr(x), f32Ptr(pre), f32Ptr(post),
	)
	return checkStatus(st, "softmax_forward")
}

func SoftmaxNormalBPF32(n, dim, seqLen, kind, rows, cols int32, temperature, entmaxAlpha float64, gumbelSeed, dtype, backend, rank int32, shape []int32, mask []byte, hier []int32, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX []float32) (loss float64, err error) {
	ut := C.int32_t(0)
	if useTarget {
		ut = 1
	}
	var closs C.double
	st := C.welvet_softmax_normalbp_f32(
		C.int32_t(n), C.int32_t(dim), C.int32_t(seqLen), C.int32_t(kind), C.int32_t(rows), C.int32_t(cols),
		C.double(temperature), C.double(entmaxAlpha), C.int32_t(gumbelSeed), C.int32_t(dtype), C.int32_t(backend),
		C.int32_t(rank), i32Ptr(shape), u8Ptr(mask), C.int32_t(len(mask)), i32Ptr(hier), C.int32_t(len(hier)),
		f32Ptr(x), f32Ptr(gradOrTarget), ut, C.double(lr), f32Ptr(pre), f32Ptr(post), f32Ptr(dX), &closs,
	)
	return float64(closs), checkStatus(st, "softmax_normalbp")
}

func KMeansForwardF32(n, featureDim, numClusters, outputMode, act int32, temperature float64, dtype, backend, rank int32, shape []int32, W, x, pre, post []float32) error {
	st := C.welvet_kmeans_forward_f32(
		C.int32_t(n), C.int32_t(featureDim), C.int32_t(numClusters), C.int32_t(outputMode), C.int32_t(act),
		C.double(temperature), C.int32_t(dtype), C.int32_t(backend), C.int32_t(rank), i32Ptr(shape),
		f32Ptr(W), f32Ptr(x), f32Ptr(pre), f32Ptr(post),
	)
	return checkStatus(st, "kmeans_forward")
}

func KMeansNormalBPF32(batch, seq, numClusters, featureDim, outputMode, act int32, temperature float64, dtype, backend int32, W, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (loss float64, err error) {
	ut := C.int32_t(0)
	if useTarget {
		ut = 1
	}
	var closs C.double
	st := C.welvet_kmeans_normalbp_f32(
		C.int32_t(batch), C.int32_t(seq), C.int32_t(numClusters), C.int32_t(featureDim), C.int32_t(outputMode),
		C.int32_t(act), C.double(temperature), C.int32_t(dtype), C.int32_t(backend),
		f32Ptr(W), f32Ptr(x), f32Ptr(gradOrTarget), ut, C.double(lr),
		f32Ptr(pre), f32Ptr(post), f32Ptr(dX), f32Ptr(dW), f32Ptr(wAfter), &closs,
	)
	return float64(closs), checkStatus(st, "kmeans_normalbp")
}

func SequentialForwardF32(batch, seq, dim, depth, childAct, dtype, backend int32, packed, x, pre, post []float32) error {
	st := C.welvet_sequential_forward_f32(
		C.int32_t(batch), C.int32_t(seq), C.int32_t(dim), C.int32_t(depth), C.int32_t(childAct),
		C.int32_t(dtype), C.int32_t(backend), f32Ptr(packed), f32Ptr(x), f32Ptr(pre), f32Ptr(post),
	)
	return checkStatus(st, "sequential_forward")
}

func SequentialNormalBPF32(batch, seq, dim, depth, childAct, dtype, backend int32, packed, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (loss float64, err error) {
	ut := C.int32_t(0)
	if useTarget {
		ut = 1
	}
	var closs C.double
	st := C.welvet_sequential_normalbp_f32(
		C.int32_t(batch), C.int32_t(seq), C.int32_t(dim), C.int32_t(depth), C.int32_t(childAct),
		C.int32_t(dtype), C.int32_t(backend), f32Ptr(packed), f32Ptr(x), f32Ptr(gradOrTarget), ut, C.double(lr),
		f32Ptr(pre), f32Ptr(post), f32Ptr(dX), f32Ptr(dW), f32Ptr(wAfter), &closs,
	)
	return float64(closs), checkStatus(st, "sequential_normalbp")
}

func ResidualForwardF32(batch, seq, dim, depth, childAct, dtype, backend int32, packed, x, pre, post []float32) error {
	st := C.welvet_residual_forward_f32(
		C.int32_t(batch), C.int32_t(seq), C.int32_t(dim), C.int32_t(depth), C.int32_t(childAct),
		C.int32_t(dtype), C.int32_t(backend), f32Ptr(packed), f32Ptr(x), f32Ptr(pre), f32Ptr(post),
	)
	return checkStatus(st, "residual_forward")
}

func ResidualNormalBPF32(batch, seq, dim, depth, childAct, dtype, backend int32, packed, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (loss float64, err error) {
	ut := C.int32_t(0)
	if useTarget {
		ut = 1
	}
	var closs C.double
	st := C.welvet_residual_normalbp_f32(
		C.int32_t(batch), C.int32_t(seq), C.int32_t(dim), C.int32_t(depth), C.int32_t(childAct),
		C.int32_t(dtype), C.int32_t(backend), f32Ptr(packed), f32Ptr(x), f32Ptr(gradOrTarget), ut, C.double(lr),
		f32Ptr(pre), f32Ptr(post), f32Ptr(dX), f32Ptr(dW), f32Ptr(wAfter), &closs,
	)
	return float64(closs), checkStatus(st, "residual_normalbp")
}

func MambaForwardF32(batch, seq, dModel, dState, expand, dtype, backend int32, packed, x, pre, post []float32) error {
	st := C.welvet_mamba_forward_f32(
		C.int32_t(batch), C.int32_t(seq), C.int32_t(dModel), C.int32_t(dState), C.int32_t(expand),
		C.int32_t(dtype), C.int32_t(backend), f32Ptr(packed), f32Ptr(x), f32Ptr(pre), f32Ptr(post),
	)
	return checkStatus(st, "mamba_forward")
}

func MambaNormalBPF32(batch, seq, dModel, dState, expand, dtype, backend int32, packed, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (loss float64, err error) {
	ut := C.int32_t(0)
	if useTarget {
		ut = 1
	}
	var closs C.double
	st := C.welvet_mamba_normalbp_f32(
		C.int32_t(batch), C.int32_t(seq), C.int32_t(dModel), C.int32_t(dState), C.int32_t(expand),
		C.int32_t(dtype), C.int32_t(backend), f32Ptr(packed), f32Ptr(x), f32Ptr(gradOrTarget), ut, C.double(lr),
		f32Ptr(pre), f32Ptr(post), f32Ptr(dX), f32Ptr(dW), f32Ptr(wAfter), &closs,
	)
	return float64(closs), checkStatus(st, "mamba_normalbp")
}

func MetacognitionForwardF32(batch, seq, dim, dtype, backend int32, packed, x, pre, post []float32) error {
	st := C.welvet_metacognition_forward_f32(
		C.int32_t(batch), C.int32_t(seq), C.int32_t(dim), C.int32_t(dtype), C.int32_t(backend),
		f32Ptr(packed), f32Ptr(x), f32Ptr(pre), f32Ptr(post),
	)
	return checkStatus(st, "metacognition_forward")
}

func MetacognitionNormalBPF32(batch, seq, dim, dtype, backend int32, packed, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (loss float64, err error) {
	ut := C.int32_t(0)
	if useTarget {
		ut = 1
	}
	var closs C.double
	st := C.welvet_metacognition_normalbp_f32(
		C.int32_t(batch), C.int32_t(seq), C.int32_t(dim), C.int32_t(dtype), C.int32_t(backend),
		f32Ptr(packed), f32Ptr(x), f32Ptr(gradOrTarget), ut, C.double(lr),
		f32Ptr(pre), f32Ptr(post), f32Ptr(dX), f32Ptr(dW), f32Ptr(wAfter), &closs,
	)
	return float64(closs), checkStatus(st, "metacognition_normalbp")
}

func GDNForwardF32(batch, seq, hidden, numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel, dtype, backend int32, packed, x, pre, post []float32) error {
	st := C.welvet_gdn_forward_f32(
		C.int32_t(batch), C.int32_t(seq), C.int32_t(hidden), C.int32_t(numKeyHeads), C.int32_t(numValueHeads),
		C.int32_t(keyHeadDim), C.int32_t(valueHeadDim), C.int32_t(convKernel), C.int32_t(dtype), C.int32_t(backend),
		f32Ptr(packed), f32Ptr(x), f32Ptr(pre), f32Ptr(post),
	)
	return checkStatus(st, "gdn_forward")
}

func GDNNormalBPF32(batch, seq, hidden, numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel, dtype, backend int32, packed, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (loss float64, err error) {
	ut := C.int32_t(0)
	if useTarget {
		ut = 1
	}
	var closs C.double
	st := C.welvet_gdn_normalbp_f32(
		C.int32_t(batch), C.int32_t(seq), C.int32_t(hidden), C.int32_t(numKeyHeads), C.int32_t(numValueHeads),
		C.int32_t(keyHeadDim), C.int32_t(valueHeadDim), C.int32_t(convKernel), C.int32_t(dtype), C.int32_t(backend),
		f32Ptr(packed), f32Ptr(x), f32Ptr(gradOrTarget), ut, C.double(lr),
		f32Ptr(pre), f32Ptr(post), f32Ptr(dX), f32Ptr(dW), f32Ptr(wAfter), &closs,
	)
	return float64(closs), checkStatus(st, "gdn_normalbp")
}

func StackNormalBPF32(batch, seq, dim, nChildren, childAct, dtype, backend int32, packed, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (loss float64, err error) {
	ut := C.int32_t(0)
	if useTarget {
		ut = 1
	}
	var closs C.double
	st := C.welvet_stack_normalbp_f32(
		C.int32_t(batch), C.int32_t(seq), C.int32_t(dim), C.int32_t(nChildren), C.int32_t(childAct),
		C.int32_t(dtype), C.int32_t(backend), f32Ptr(packed), f32Ptr(x), f32Ptr(gradOrTarget), ut, C.double(lr),
		f32Ptr(pre), f32Ptr(post), f32Ptr(dX), f32Ptr(dW), f32Ptr(wAfter), &closs,
	)
	return float64(closs), checkStatus(st, "stack_normalbp")
}

func ParallelForwardF32(batch, seq, dim, outFeat, branches, combine, sparseK, dtype, backend int32, disagreeBeta float64, packed, x, pre, post []float32) error {
	st := C.welvet_parallel_forward_f32(
		C.int32_t(batch), C.int32_t(seq), C.int32_t(dim), C.int32_t(outFeat), C.int32_t(branches),
		C.int32_t(combine), C.int32_t(sparseK), C.int32_t(dtype), C.int32_t(backend), C.double(disagreeBeta),
		f32Ptr(packed), f32Ptr(x), f32Ptr(pre), f32Ptr(post),
	)
	return checkStatus(st, "parallel_forward")
}

func ParallelNormalBPF32(batch, seq, dim, outFeat, branches, combine, sparseK, dtype, backend int32, disagreeBeta float64, packed, x, gradOrTarget []float32, useTarget bool, lr float64, pre, post, dX, dW, wAfter []float32) (loss float64, err error) {
	ut := C.int32_t(0)
	if useTarget {
		ut = 1
	}
	var closs C.double
	st := C.welvet_parallel_normalbp_f32(
		C.int32_t(batch), C.int32_t(seq), C.int32_t(dim), C.int32_t(outFeat), C.int32_t(branches),
		C.int32_t(combine), C.int32_t(sparseK), C.int32_t(dtype), C.int32_t(backend), C.double(disagreeBeta),
		f32Ptr(packed), f32Ptr(x), f32Ptr(gradOrTarget), ut, C.double(lr),
		f32Ptr(pre), f32Ptr(post), f32Ptr(dX), f32Ptr(dW), f32Ptr(wAfter), &closs,
	)
	return float64(closs), checkStatus(st, "parallel_normalbp")
}

func StackStepBPDenseF32(nChildren, nTicks, batch int32, lr float64, dtype int32, inH, outH, act []int32, packedW, xTicks, targetTicks, wAfter []float32) (loss float64, err error) {
	var closs C.double
	st := C.welvet_stack_stepbp_dense_f32(
		C.int32_t(nChildren), C.int32_t(nTicks), C.int32_t(batch), C.double(lr), C.int32_t(dtype),
		i32Ptr(inH), i32Ptr(outH), i32Ptr(act), f32Ptr(packedW), f32Ptr(xTicks), f32Ptr(targetTicks),
		f32Ptr(wAfter), &closs,
	)
	return float64(closs), checkStatus(st, "stack_stepbp_dense")
}

func StackTrainMSE(mode, batch, seq, dim, nChildren, childAct, dtype, backend int32, packed, x, target []float32, lr float64, wAfter []float32) (loss float64, err error) {
	var closs C.double
	st := C.welvet_stack_train_mse(
		C.int32_t(mode), C.int32_t(batch), C.int32_t(seq), C.int32_t(dim), C.int32_t(nChildren),
		C.int32_t(childAct), C.int32_t(dtype), C.int32_t(backend),
		f32Ptr(packed), f32Ptr(x), f32Ptr(target), C.double(lr), f32Ptr(wAfter), &closs,
	)
	return float64(closs), checkStatus(st, "stack_train_mse")
}

func ParallelTrainMSE(mode, batch, seq, dim, outFeat, branches, combine, sparseK, dtype, backend int32, disagreeBeta float64, packed, x, target []float32, lr float64, wAfter []float32) (loss float64, err error) {
	var closs C.double
	st := C.welvet_parallel_train_mse(
		C.int32_t(mode), C.int32_t(batch), C.int32_t(seq), C.int32_t(dim), C.int32_t(outFeat),
		C.int32_t(branches), C.int32_t(combine), C.int32_t(sparseK), C.int32_t(dtype), C.int32_t(backend),
		C.double(disagreeBeta), f32Ptr(packed), f32Ptr(x), f32Ptr(target), C.double(lr),
		f32Ptr(wAfter), &closs,
	)
	return float64(closs), checkStatus(st, "parallel_train_mse")
}

// GridTrainMSE runs mesh-clock TrainMSE (welvet_grid_train_mse).
// cellKind: 1=Dense, 19=Stack, 23=Parallel (OpKind). nSub = NestedStack children or Parallel branches.
func GridTrainMSE(mode, depth, rows, cols, lpc, batch, ticks, dim, cellKind, nSub, childAct, dtype, backend int32, packed, x, target []float32, lr float64, wAfter []float32) (loss float64, err error) {
	var closs C.double
	st := C.welvet_grid_train_mse(
		C.int32_t(mode), C.int32_t(depth), C.int32_t(rows), C.int32_t(cols), C.int32_t(lpc),
		C.int32_t(batch), C.int32_t(ticks), C.int32_t(dim), C.int32_t(cellKind), C.int32_t(nSub),
		C.int32_t(childAct), C.int32_t(dtype), C.int32_t(backend),
		f32Ptr(packed), f32Ptr(x), f32Ptr(target), C.double(lr), f32Ptr(wAfter), &closs,
	)
	return float64(closs), checkStatus(st, "grid_train_mse")
}

// StackLineTrainMSE runs systolic Line TrainMSE by TrainMode (welvet_stack_line_train_mse).
func StackLineTrainMSE(mode, nChildren, nTicks, batch int32, lr float64, dtype int32, inH, outH, act []int32, packedW, xTicks, targetTicks, wAfter []float32) (loss float64, err error) {
	var closs C.double
	st := C.welvet_stack_line_train_mse(
		C.int32_t(mode), C.int32_t(nChildren), C.int32_t(nTicks), C.int32_t(batch), C.double(lr),
		C.int32_t(dtype), i32Ptr(inH), i32Ptr(outH), i32Ptr(act), f32Ptr(packedW),
		f32Ptr(xTicks), f32Ptr(targetTicks), f32Ptr(wAfter), &closs,
	)
	return float64(closs), checkStatus(st, "stack_line_train_mse")
}

func SeqMixKindName(kind int32) (string, error) {
	buf := make([]byte, 64)
	st := C.welvet_seqmix_kind_name(C.int32_t(kind), (*C.char)(unsafe.Pointer(&buf[0])), C.int32_t(len(buf)))
	if err := checkStatus(st, "seqmix_kind_name"); err != nil {
		return "", err
	}
	n := 0
	for n < len(buf) && buf[n] != 0 {
		n++
	}
	return string(buf[:n]), nil
}

func SeqMixMHAKind() int32   { return int32(C.welvet_seqmix_mha_kind()) }
func SeqMixMambaKind() int32 { return int32(C.welvet_seqmix_mamba_kind()) }
func SeqMixGDNKind() int32   { return int32(C.welvet_seqmix_gdn_kind()) }
