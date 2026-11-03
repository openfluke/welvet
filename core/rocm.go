package core

import (
	"fmt"
	"sync"
)

// RocmExecutor is installed by nativerocm.EnableRouting. Keeping this contract
// in core avoids a layer ↔ native binding import cycle. All numerical work is
// performed by the independent native ROCm backend.
type RocmExecutor interface {
	TrainCredit(any, *Tensor[float32], *Tensor[float32], *Tensor[float32], int, float64) error
	DenseProxy(any, *Tensor[float32], *Tensor[float32], *Tensor[float32], int) (*Tensor[float32], error)
	Forward(any, *Tensor[float32]) (*Tensor[float32], *Tensor[float32], error)
	Backward(any, *Tensor[float32], *Tensor[float32]) (*Tensor[float32], *Tensor[float32], error)
	ApplySGD(any, *Tensor[float32], float64) error
	TrainMSE(any, *Tensor[float32], *Tensor[float32], int, float64) (float64, error)
}

var rocmRoute struct {
	sync.RWMutex
	executor RocmExecutor
}

// RegisterRocmExecutor returns an unregister function. Only one router can own
// process-wide automatic placement at a time; explicit sessions are independent.
func RegisterRocmExecutor(e RocmExecutor) (func(), error) {
	rocmRoute.Lock()
	defer rocmRoute.Unlock()
	if e == nil || rocmRoute.executor != nil {
		return nil, fmt.Errorf("rocm: routing already registered or nil executor")
	}
	rocmRoute.executor = e
	return func() {
		rocmRoute.Lock()
		defer rocmRoute.Unlock()
		if rocmRoute.executor == e {
			rocmRoute.executor = nil
		}
	}, nil
}
func RocmRoutingAvailable() bool {
	rocmRoute.RLock()
	defer rocmRoute.RUnlock()
	return rocmRoute.executor != nil
}
func rocmInput[T Numeric](t *Tensor[T]) (*Tensor[float32], error) {
	v, ok := any(t).(*Tensor[float32])
	if !ok || v == nil {
		return nil, fmt.Errorf("rocm: routing requires Float32 activation tensors")
	}
	return v, nil
}
func rocmResult[T Numeric](t *Tensor[float32]) *Tensor[T] {
	if t == nil {
		return nil
	}
	return any(t).(*Tensor[T])
}
func RocmForward[T Numeric](op any, x *Tensor[T]) (*Tensor[T], *Tensor[T], error) {
	v, err := rocmInput(x)
	if err != nil {
		return nil, nil, err
	}
	rocmRoute.RLock()
	defer rocmRoute.RUnlock()
	if rocmRoute.executor == nil {
		return nil, nil, fmt.Errorf("rocm: call nativerocm.EnableRouting before selecting BackendRocm")
	}
	pre, post, err := rocmRoute.executor.Forward(op, v)
	return rocmResult[T](pre), rocmResult[T](post), err
}
func RocmBackward[T Numeric](op any, gy, x *Tensor[T]) (*Tensor[T], *Tensor[T], error) {
	v, err := rocmInput(x)
	if err != nil {
		return nil, nil, err
	}
	g, err := rocmInput(gy)
	if err != nil {
		return nil, nil, err
	}
	rocmRoute.RLock()
	defer rocmRoute.RUnlock()
	if rocmRoute.executor == nil {
		return nil, nil, fmt.Errorf("rocm: routing is not enabled")
	}
	gx, dw, err := rocmRoute.executor.Backward(op, g, v)
	return rocmResult[T](gx), rocmResult[T](dw), err
}
func RocmApplySGD[T Numeric](op any, dw *Tensor[T], lr float64) error {
	rocmRoute.RLock()
	defer rocmRoute.RUnlock()
	if rocmRoute.executor == nil {
		return fmt.Errorf("rocm: routing is not enabled")
	}
	if dw == nil || len(dw.Data) == 0 {
		return nil
	}
	v, err := rocmInput(dw)
	if err != nil {
		return err
	}
	return rocmRoute.executor.ApplySGD(op, v, lr)
}
func RocmTrainMSE[T Numeric](op any, x, target *Tensor[T], mode int, lr float64) (float64, error) {
	v, err := rocmInput(x)
	if err != nil {
		return 0, err
	}
	t, err := rocmInput(target)
	if err != nil {
		return 0, err
	}
	rocmRoute.RLock()
	defer rocmRoute.RUnlock()
	if rocmRoute.executor == nil {
		return 0, fmt.Errorf("rocm: routing is not enabled")
	}
	return rocmRoute.executor.TrainMSE(op, v, t, mode, lr)
}

func RocmDenseProxy[T Numeric](op any, gy, x, pre *Tensor[T], kind int) (*Tensor[T], error) {
	v, err := rocmInput(x)
	if err != nil {
		return nil, err
	}
	g, err := rocmInput(gy)
	if err != nil {
		return nil, err
	}
	var p *Tensor[float32]
	if pre != nil {
		p, err = rocmInput(pre)
		if err != nil {
			return nil, err
		}
	}
	rocmRoute.RLock()
	defer rocmRoute.RUnlock()
	if rocmRoute.executor == nil {
		return nil, fmt.Errorf("rocm: routing is not enabled")
	}
	out, err := rocmRoute.executor.DenseProxy(op, g, v, p, kind)
	return rocmResult[T](out), err
}

func RocmTrainCredit[T Numeric](op any, gy, x, pre *Tensor[T], mode int, lr float64) error {
	v, err := rocmInput(x)
	if err != nil {
		return err
	}
	g, err := rocmInput(gy)
	if err != nil {
		return err
	}
	var p *Tensor[float32]
	if pre != nil {
		p, err = rocmInput(pre)
		if err != nil {
			return err
		}
	}
	rocmRoute.RLock()
	defer rocmRoute.RUnlock()
	if rocmRoute.executor == nil {
		return fmt.Errorf("rocm: routing is not enabled")
	}
	return rocmRoute.executor.TrainCredit(op, g, v, p, mode, lr)
}
