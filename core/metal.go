package core

import (
	"fmt"
	"sync"
)

// MetalExecutor is installed by nativemetal.EnableRouting. Keeping this contract
// in core avoids a layer ↔ native binding import cycle. All numerical work is
// performed by the independent native Metal backend.
type MetalExecutor interface {
	TrainCredit(any, *Tensor[float32], *Tensor[float32], *Tensor[float32], int, float64) error
	DenseProxy(any, *Tensor[float32], *Tensor[float32], *Tensor[float32], int) (*Tensor[float32], error)
	Forward(any, *Tensor[float32]) (*Tensor[float32], *Tensor[float32], error)
	Backward(any, *Tensor[float32], *Tensor[float32]) (*Tensor[float32], *Tensor[float32], error)
	ApplySGD(any, *Tensor[float32], float64) error
	TrainMSE(any, *Tensor[float32], *Tensor[float32], int, float64) (float64, error)
}

var metalRoute struct {
	sync.RWMutex
	executor MetalExecutor
}

// RegisterMetalExecutor returns an unregister function. Only one router can own
// process-wide automatic placement at a time; explicit sessions are independent.
func RegisterMetalExecutor(e MetalExecutor) (func(), error) {
	metalRoute.Lock()
	defer metalRoute.Unlock()
	if e == nil || metalRoute.executor != nil {
		return nil, fmt.Errorf("metal: routing already registered or nil executor")
	}
	metalRoute.executor = e
	return func() {
		metalRoute.Lock()
		defer metalRoute.Unlock()
		if metalRoute.executor == e {
			metalRoute.executor = nil
		}
	}, nil
}
func MetalRoutingAvailable() bool {
	metalRoute.RLock()
	defer metalRoute.RUnlock()
	return metalRoute.executor != nil
}
func metalInput[T Numeric](t *Tensor[T]) (*Tensor[float32], error) {
	v, ok := any(t).(*Tensor[float32])
	if !ok || v == nil {
		return nil, fmt.Errorf("metal: routing requires Float32 activation tensors")
	}
	return v, nil
}
func metalResult[T Numeric](t *Tensor[float32]) *Tensor[T] {
	if t == nil {
		return nil
	}
	return any(t).(*Tensor[T])
}
func MetalForward[T Numeric](op any, x *Tensor[T]) (*Tensor[T], *Tensor[T], error) {
	v, err := metalInput(x)
	if err != nil {
		return nil, nil, err
	}
	metalRoute.RLock()
	defer metalRoute.RUnlock()
	if metalRoute.executor == nil {
		return nil, nil, fmt.Errorf("metal: call nativemetal.EnableRouting before selecting BackendMetal")
	}
	pre, post, err := metalRoute.executor.Forward(op, v)
	return metalResult[T](pre), metalResult[T](post), err
}
func MetalBackward[T Numeric](op any, gy, x *Tensor[T]) (*Tensor[T], *Tensor[T], error) {
	v, err := metalInput(x)
	if err != nil {
		return nil, nil, err
	}
	g, err := metalInput(gy)
	if err != nil {
		return nil, nil, err
	}
	metalRoute.RLock()
	defer metalRoute.RUnlock()
	if metalRoute.executor == nil {
		return nil, nil, fmt.Errorf("metal: routing is not enabled")
	}
	gx, dw, err := metalRoute.executor.Backward(op, g, v)
	return metalResult[T](gx), metalResult[T](dw), err
}
func MetalApplySGD[T Numeric](op any, dw *Tensor[T], lr float64) error {
	metalRoute.RLock()
	defer metalRoute.RUnlock()
	if metalRoute.executor == nil {
		return fmt.Errorf("metal: routing is not enabled")
	}
	if dw == nil || len(dw.Data) == 0 {
		return nil
	}
	v, err := metalInput(dw)
	if err != nil {
		return err
	}
	return metalRoute.executor.ApplySGD(op, v, lr)
}
func MetalTrainMSE[T Numeric](op any, x, target *Tensor[T], mode int, lr float64) (float64, error) {
	v, err := metalInput(x)
	if err != nil {
		return 0, err
	}
	t, err := metalInput(target)
	if err != nil {
		return 0, err
	}
	metalRoute.RLock()
	defer metalRoute.RUnlock()
	if metalRoute.executor == nil {
		return 0, fmt.Errorf("metal: routing is not enabled")
	}
	return metalRoute.executor.TrainMSE(op, v, t, mode, lr)
}

// InheritExec keeps an explicitly assigned child placement. An inherited parent
// pin is cleared so grandchildren can still receive their own JSON assignments.
func InheritExec(current, parent ExecConfig) ExecConfig {
	if current.BackendPinned {
		return current
	}
	parent.BackendPinned = false
	return parent
}

func MetalDenseProxy[T Numeric](op any, gy, x, pre *Tensor[T], kind int) (*Tensor[T], error) {
	v, err := metalInput(x)
	if err != nil {
		return nil, err
	}
	g, err := metalInput(gy)
	if err != nil {
		return nil, err
	}
	var p *Tensor[float32]
	if pre != nil {
		p, err = metalInput(pre)
		if err != nil {
			return nil, err
		}
	}
	metalRoute.RLock()
	defer metalRoute.RUnlock()
	if metalRoute.executor == nil {
		return nil, fmt.Errorf("metal: routing is not enabled")
	}
	out, err := metalRoute.executor.DenseProxy(op, g, v, p, kind)
	return metalResult[T](out), err
}

func MetalTrainCredit[T Numeric](op any, gy, x, pre *Tensor[T], mode int, lr float64) error {
	v, err := metalInput(x)
	if err != nil {
		return err
	}
	g, err := metalInput(gy)
	if err != nil {
		return err
	}
	var p *Tensor[float32]
	if pre != nil {
		p, err = metalInput(pre)
		if err != nil {
			return err
		}
	}
	metalRoute.RLock()
	defer metalRoute.RUnlock()
	if metalRoute.executor == nil {
		return fmt.Errorf("metal: routing is not enabled")
	}
	return metalRoute.executor.TrainCredit(op, g, v, p, mode, lr)
}
