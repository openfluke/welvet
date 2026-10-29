package core

import (
	"fmt"
	"sync"
)

// CudaExecutor is installed by nativecuda.EnableRouting. Keeping this contract
// in core avoids a layer ↔ native binding import cycle. All numerical work is
// performed by the independent native CUDA backend.
type CudaExecutor interface {
	TrainCredit(any, *Tensor[float32], *Tensor[float32], *Tensor[float32], int, float64) error
	DenseProxy(any, *Tensor[float32], *Tensor[float32], *Tensor[float32], int) (*Tensor[float32], error)
	Forward(any, *Tensor[float32]) (*Tensor[float32], *Tensor[float32], error)
	Backward(any, *Tensor[float32], *Tensor[float32]) (*Tensor[float32], *Tensor[float32], error)
	ApplySGD(any, *Tensor[float32], float64) error
	TrainMSE(any, *Tensor[float32], *Tensor[float32], int, float64) (float64, error)
}

var cudaRoute struct {
	sync.RWMutex
	executor CudaExecutor
}

// RegisterCudaExecutor returns an unregister function. Only one router can own
// process-wide automatic placement at a time; explicit sessions are independent.
func RegisterCudaExecutor(e CudaExecutor) (func(), error) {
	cudaRoute.Lock()
	defer cudaRoute.Unlock()
	if e == nil || cudaRoute.executor != nil {
		return nil, fmt.Errorf("cuda: routing already registered or nil executor")
	}
	cudaRoute.executor = e
	return func() {
		cudaRoute.Lock()
		defer cudaRoute.Unlock()
		if cudaRoute.executor == e {
			cudaRoute.executor = nil
		}
	}, nil
}
func CudaRoutingAvailable() bool {
	cudaRoute.RLock()
	defer cudaRoute.RUnlock()
	return cudaRoute.executor != nil
}
func cudaInput[T Numeric](t *Tensor[T]) (*Tensor[float32], error) {
	v, ok := any(t).(*Tensor[float32])
	if !ok || v == nil {
		return nil, fmt.Errorf("cuda: routing requires Float32 activation tensors")
	}
	return v, nil
}
func cudaResult[T Numeric](t *Tensor[float32]) *Tensor[T] {
	if t == nil {
		return nil
	}
	return any(t).(*Tensor[T])
}
func CudaForward[T Numeric](op any, x *Tensor[T]) (*Tensor[T], *Tensor[T], error) {
	v, err := cudaInput(x)
	if err != nil {
		return nil, nil, err
	}
	cudaRoute.RLock()
	defer cudaRoute.RUnlock()
	if cudaRoute.executor == nil {
		return nil, nil, fmt.Errorf("cuda: call nativecuda.EnableRouting before selecting BackendCuda")
	}
	pre, post, err := cudaRoute.executor.Forward(op, v)
	return cudaResult[T](pre), cudaResult[T](post), err
}
func CudaBackward[T Numeric](op any, gy, x *Tensor[T]) (*Tensor[T], *Tensor[T], error) {
	v, err := cudaInput(x)
	if err != nil {
		return nil, nil, err
	}
	g, err := cudaInput(gy)
	if err != nil {
		return nil, nil, err
	}
	cudaRoute.RLock()
	defer cudaRoute.RUnlock()
	if cudaRoute.executor == nil {
		return nil, nil, fmt.Errorf("cuda: routing is not enabled")
	}
	gx, dw, err := cudaRoute.executor.Backward(op, g, v)
	return cudaResult[T](gx), cudaResult[T](dw), err
}
func CudaApplySGD[T Numeric](op any, dw *Tensor[T], lr float64) error {
	cudaRoute.RLock()
	defer cudaRoute.RUnlock()
	if cudaRoute.executor == nil {
		return fmt.Errorf("cuda: routing is not enabled")
	}
	if dw == nil || len(dw.Data) == 0 {
		return nil
	}
	v, err := cudaInput(dw)
	if err != nil {
		return err
	}
	return cudaRoute.executor.ApplySGD(op, v, lr)
}
func CudaTrainMSE[T Numeric](op any, x, target *Tensor[T], mode int, lr float64) (float64, error) {
	v, err := cudaInput(x)
	if err != nil {
		return 0, err
	}
	t, err := cudaInput(target)
	if err != nil {
		return 0, err
	}
	cudaRoute.RLock()
	defer cudaRoute.RUnlock()
	if cudaRoute.executor == nil {
		return 0, fmt.Errorf("cuda: routing is not enabled")
	}
	return cudaRoute.executor.TrainMSE(op, v, t, mode, lr)
}

func CudaDenseProxy[T Numeric](op any, gy, x, pre *Tensor[T], kind int) (*Tensor[T], error) {
	v, err := cudaInput(x)
	if err != nil {
		return nil, err
	}
	g, err := cudaInput(gy)
	if err != nil {
		return nil, err
	}
	var p *Tensor[float32]
	if pre != nil {
		p, err = cudaInput(pre)
		if err != nil {
			return nil, err
		}
	}
	cudaRoute.RLock()
	defer cudaRoute.RUnlock()
	if cudaRoute.executor == nil {
		return nil, fmt.Errorf("cuda: routing is not enabled")
	}
	out, err := cudaRoute.executor.DenseProxy(op, g, v, p, kind)
	return cudaResult[T](out), err
}

func CudaTrainCredit[T Numeric](op any, gy, x, pre *Tensor[T], mode int, lr float64) error {
	v, err := cudaInput(x)
	if err != nil {
		return err
	}
	g, err := cudaInput(gy)
	if err != nil {
		return err
	}
	var p *Tensor[float32]
	if pre != nil {
		p, err = cudaInput(pre)
		if err != nil {
			return err
		}
	}
	cudaRoute.RLock()
	defer cudaRoute.RUnlock()
	if cudaRoute.executor == nil {
		return fmt.Errorf("cuda: routing is not enabled")
	}
	return cudaRoute.executor.TrainCredit(op, g, v, p, mode, lr)
}
